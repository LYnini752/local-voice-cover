"""Local audio pipeline. Model inference runs in an isolated Python process."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / ".local" / "jobs"
SEED_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"
REFERENCE_LIMIT = 25
SOURCE_LIMIT = 60


def ffmpeg_path():
    configured = os.environ.get("FFMPEG_BINARY")
    if configured:
        path = shutil.which(configured)
        if not path:
            raise ValueError("FFMPEG_BINARY 指向的程序不存在")
        return path
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        raise ValueError("未找到 FFmpeg，请安装 requirements.txt 中的 imageio-ffmpeg") from None


def engine_paths():
    engine = Path(os.getenv("SEED_VC_DIR", ROOT / "vendor" / "seed-vc")).expanduser().resolve()
    python = Path(os.getenv("SEED_VC_PYTHON", ROOT / ".venv-engine" / "bin" / "python")).expanduser().absolute()
    if not (engine / "inference.py").is_file() or not python.is_file():
        raise ValueError("模型引擎尚未安装，请运行 python scripts/setup_engine.py")
    return engine, python


def validate_options(steps, shift, voice_gain, music_gain, mode, device):
    for value, low, high, name in [(steps, 10, 100, "转换步数"), (shift, -12, 12, "移调")]:
        if isinstance(value, bool) or not math.isfinite(float(value)) or int(value) != float(value) or not low <= value <= high:
            raise ValueError(f"{name}必须是 {low}–{high} 的整数")
    for value in (voice_gain, music_gain):
        if not math.isfinite(float(value)) or not 0 <= float(value) <= 2:
            raise ValueError("混音音量必须是 0–2 之间的有限数值")
    if mode not in {"song", "vocals"}:
        raise ValueError("输入类型必须是 song 或 vocals")
    if device not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("无效的计算设备")


def seed_command(python, source, reference, output, steps, shift, engine=None, device="auto"):
    validate_options(steps, shift, 1, 1, "vocals", device)
    return [str(python), str(ROOT / "seed_runner.py"), "--engine-dir", str(engine or ROOT / "vendor" / "seed-vc"),
            "--source", str(source), "--target", str(reference), "--output", str(output),
            "--diffusion-steps", str(int(steps)), "--semi-tone-shift", str(int(shift)),
            "--f0-condition", "True", "--device", device]


def execute(args, cwd, log, timeout=1800):
    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["PYTHONUNBUFFERED"] = "1"
    with Path(log).open("a", encoding="utf-8") as stream:
        try:
            subprocess.run([str(a) for a in args], cwd=cwd, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"单阶段处理超过 {timeout // 60} 分钟，已停止。查看任务 process.log。") from None
        except subprocess.CalledProcessError:
            raise RuntimeError("处理失败。查看任务 process.log；设备内存不足时可选择 CPU 后重试。") from None


@contextmanager
def inference_lock():
    """Prevent two UI/CLI processes from loading large models simultaneously."""
    import fcntl
    JOBS.mkdir(parents=True, exist_ok=True)
    with (JOBS.parent / "inference.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("已有转换任务正在运行，请等待它结束。") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def check_file(path):
    p = Path(path).expanduser().resolve()
    if not p.is_file() or p.stat().st_size == 0:
        raise ValueError("输入文件不存在或为空")
    if p.stat().st_size > 500 * 1024 * 1024:
        raise ValueError("单个音频文件不能超过 500 MB")
    return p


def audio_stats(path, minimum=0.5, require_signal=True):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if len(audio) / rate < minimum:
        raise ValueError(f"音频过短，至少需要 {minimum} 秒")
    if not np.isfinite(audio).all():
        raise ValueError("音频包含无效采样值")
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if require_signal and (peak < 1e-5 or rms < 1e-6):
        raise ValueError("音频接近静音，请使用清晰可听的音频")
    return {"seconds": round(len(audio) / rate, 3), "sample_rate": rate,
            "channels": audio.shape[1], "peak": round(peak, 6), "rms": round(rms, 6)}


def normalize(ffmpeg, source, output, seconds, channels, log):
    execute([ffmpeg, "-hide_banner", "-nostdin", "-y", "-i", source,
             "-t", str(seconds), "-vn", "-ac", str(channels), "-ar", "44100", "-c:a", "pcm_f32le", output], ROOT, log, timeout=120)
    return audio_stats(output)


def mix_audio(ffmpeg, vocal, music, output, voice_gain, music_gain, log):
    # Uncompensated limiter lookahead can shift the output; latency=1 preserves alignment.
    filters = (f"[0:a]aformat=channel_layouts=stereo,volume={float(voice_gain)}[v];"
               f"[1:a]aformat=channel_layouts=stereo,volume={float(music_gain)}[m];"
               "[v][m]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95:level=0:latency=1[out]")
    execute([ffmpeg, "-hide_banner", "-nostdin", "-y", "-i", vocal, "-i", music,
             "-filter_complex", filters, "-map", "[out]", "-ar", "44100", "-c:a", "pcm_s16le", output], ROOT, log, timeout=120)


def _convert(reference, source, accompaniment, steps, shift, voice_gain, music_gain, mode, device, progress):
    if not reference or not source:
        raise ValueError("需要参考声音和目标歌曲两段音频")
    validate_options(steps, shift, voice_gain, music_gain, mode, device)
    reference, source = check_file(reference), check_file(source)
    accompaniment = check_file(accompaniment) if accompaniment else None
    if mode == "song" and accompaniment:
        raise ValueError("整曲模式会自动提取伴奏；另传伴奏时请选择纯人声模式")
    seed, python = engine_paths()
    ffmpeg = ffmpeg_path()
    job = JOBS / uuid.uuid4().hex
    job.mkdir(parents=True)
    log = job / "process.log"
    metadata_path = job / "metadata.json"
    start = time.monotonic()
    metadata = {"job_id": job.name, "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "running", "engine": "Seed-VC", "expected_engine_commit": SEED_COMMIT,
                "steps": int(steps), "pitch_shift": int(shift), "input_mode": mode, "device_requested": device,
                "source_limit_seconds": SOURCE_LIMIT, "reference_limit_seconds": REFERENCE_LIMIT,
                "synthetic_audio": True, "voice_gain": float(voice_gain), "music_gain": float(music_gain)}

    def stage(name, fraction):
        metadata["stage"] = name
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if progress:
            progress(fraction, desc=name)

    try:
        stage("准备音频", 0.05)
        metadata["reference"] = normalize(ffmpeg, reference, job / "reference.wav", REFERENCE_LIMIT, 1, log)
        metadata["source"] = normalize(ffmpeg, source, job / "source.wav", SOURCE_LIMIT, 2 if mode == "song" else 1, log)
        if metadata["reference"]["seconds"] < 3:
            raise ValueError("参考声音至少需要 3 秒，建议使用 6–25 秒干净独唱或说话")
        source_vocal = job / "source.wav"
        music = None
        if mode == "song":
            stage("分离歌曲人声和伴奏（首次需下载模型）", 0.15)
            execute([python, ROOT / "model_cache.py"], ROOT, log)
            execute([python, "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs", "-d", "cpu",
                     "--shifts", "1", "--segment", "7", "--jobs", "1", "--float32", "--out", job / "stems", job / "source.wav"], ROOT, log)
            stems = job / "stems" / "htdemucs" / "source"
            source_vocal = stems / "vocals.wav"
            music = stems / "no_vocals.wav"
            audio_stats(source_vocal)
            if not music.is_file():
                raise RuntimeError("人声分离没有生成伴奏")
        elif accompaniment:
            music = job / "accompaniment.wav"
            normalize(ffmpeg, accompaniment, music, SOURCE_LIMIT, 2, log)
            if sf.info(music).duration + 0.05 < metadata["source"]["seconds"]:
                raise ValueError("伴奏短于目标人声，请提供从同一时间点开始且长度足够的伴奏")
        stage("转换音色（首次需下载模型）", 0.4)
        out = job / "converted"
        out.mkdir()
        execute(seed_command(python, source_vocal, job / "reference.wav", out, steps, shift, seed, device), seed, log)
        candidates = list(out.glob("*.wav"))
        if len(candidates) != 1:
            raise RuntimeError("模型没有生成唯一的 WAV 输出")
        vocal = job / "vocal.wav"
        shutil.copyfile(candidates[0], vocal)
        metadata["vocal"] = audio_stats(vocal)
        if abs(metadata["vocal"]["seconds"] - metadata["source"]["seconds"]) > 0.5:
            raise RuntimeError("转换结果时长异常，与输入相差超过 0.5 秒")
        stage("导出音频", 0.9)
        result = job / "result.wav"
        if music:
            mix_audio(ffmpeg, vocal, music, result, voice_gain, music_gain, log)
        else:
            execute([ffmpeg, "-hide_banner", "-nostdin", "-y", "-i", vocal, "-af",
                     f"volume={float(voice_gain)},alimiter=limit=0.95:level=0:latency=1", "-c:a", "pcm_s16le", result], ROOT, log, timeout=120)
        metadata["output"] = audio_stats(result, require_signal=False)
        metadata["output_sha256"] = hashlib.sha256(result.read_bytes()).hexdigest()
        metadata["model_revisions"] = {}
        for cache in (seed / "checkpoints", seed / "checkpoints" / "hf_cache"):
            for ref in cache.glob("models--*/refs/main"):
                metadata["model_revisions"][ref.parent.parent.name.removeprefix("models--").replace("--", "/")] = ref.read_text().strip()
        metadata["status"] = "complete"
        stage("完成", 1.0)
        return str(vocal), str(result), str(metadata_path)
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = str(exc)
        raise RuntimeError(f"{exc}\n任务：{job.name}；日志：{log}") from None
    finally:
        metadata["elapsed_seconds"] = round(time.monotonic() - start, 2)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def convert(reference, source, accompaniment=None, steps=30, shift=0, voice_gain=1.0, music_gain=0.8,
            mode="vocals", device="auto", progress=None):
    with inference_lock():
        return _convert(reference, source, accompaniment, steps, shift, voice_gain, music_gain, mode, device, progress)
