# Local Voice Cover

[简体中文](README.md) | [English](README.en.md)

Turn **two audio inputs — a reference voice and a target song — into a local AI cover**. The app separates vocals, converts their timbre using the reference, and mixes the converted vocals with the accompaniment. No per-speaker training is required.

This is a short-clip application built on [Seed-VC](https://github.com/Plachtaa/seed-vc) and [Demucs](https://github.com/facebookresearch/demucs), not a newly trained foundation model. **Software alpha, not real-time.** See [VALIDATION.md](VALIDATION.md) for actual runs and limitations; the validation document is currently in Chinese.

## Features and limits

- Song mode: vocal separation → voice conversion → accompaniment remix.
- Vocals mode: convert isolated vocals, optionally adding accompaniment aligned to the same starting time.
- Only the first **25 seconds of the reference** and **60 seconds of the source** are processed. The reference must be at least 3 seconds; 6–25 seconds of clear solo speech or singing is recommended.
- Adjustable inference steps, vocal pitch shift in semitones, and vocal/accompaniment volume. Download WAV audio and a JSON processing record.
- Local web interface and CLI, progress reporting, saved error logs, and a single-job lock.
- Apple Silicon adaptation: float32 pitch processing, diffusion on MPS, and BigVGAN decoding on CPU.

Voice similarity, pronunciation, and musical quality depend on the input. Background music in the reference, overlapping speakers, and large pitch differences can reduce quality. Pitch shifting changes the converted vocals only, not the accompaniment. This version does not process unlimited-length songs.

## Install and run

Requires Git and Python 3.11–3.13 on macOS/Linux. The installer creates a **separate Python 3.10 model environment**. Initial dependency/model downloads need internet access and approximately 10 GB of free disk space. Start with a 10-second clip on a computer with 16 GB RAM; this is a starting suggestion, not a performance guarantee.

```bash
git clone https://github.com/LYnini752/local-voice-cover.git
cd local-voice-cover
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt uv==0.13.9
python scripts/setup_engine.py
python scripts/setup_engine.py --check
python cli.py --doctor
python model_cache.py  # Optional: pre-download the vocal-separation model
python app.py
```

Open **http://127.0.0.1:7862**. Upload a reference voice and source audio, select song or vocals mode, then click **开始转换** (Start conversion). Listen to and download the result. The interface currently uses Chinese labels; this English guide does not imply an English UI.

The installer pins the Seed-VC revision and primary dependencies and preserves existing engine modifications. Do not install `requirements-engine.txt` into the UI environment. `requirements-engine.lock.txt` records the tested Mac environment; use the primary requirements file for cross-platform installation.

## Command line

```bash
python cli.py --reference '/path/to/reference.wav' \
  --source '/path/to/song.mp3' --mode song --steps 30

python cli.py --reference '/path/to/reference.wav' \
  --source '/path/to/vocals.wav' --mode vocals --device cpu
```

Add `--accompaniment '/path/to/music.wav'` in vocals mode to mix another track. Quote paths containing spaces. `--shift -2` lowers the converted vocals by two semitones.

| Environment variable | Default / purpose |
|---|---|
| `SEED_VC_DIR` | `vendor/seed-vc` in this repository; override for a compatible engine |
| `SEED_VC_PYTHON` | `.venv-engine/bin/python`; override with an absolute interpreter path |
| `FFMPEG_BINARY` | System FFmpeg or the imageio-provided binary |
| `PORT` | Web port, default `7862` |
| `HF_HUB_OFFLINE` | Set to `1` only after Hugging Face model caches are complete |

Automatic device selection prefers CUDA, then Apple MPS, then CPU. Separation runs on CPU. Hardware support in code does not mean every Linux/CUDA/CPU configuration has been tested; consult the validation record.

## Outputs and privacy

Each job writes to `.local/jobs/<job-id>/`:

- `vocal.wav`: converted vocals; `result.wav`: final AI-generated audio.
- `metadata.json`: status, settings, durations, and output hashes.
- `process.log`: model logs and failure details.
- Normalized inputs and intermediate/separated audio.

This application does not upload input audio to an inference service. Gradio analytics are disabled; model installation/downloads contact GitHub, Hugging Face, and Demucs model hosts. The UI binds to loopback only and does not create a public sharing link. Gradio may also retain uploaded files in the system temporary directory.

Jobs are not automatically deleted. After saving your outputs, delete unwanted job folders using your file manager; do not delete an active job. Conversion checkpoints live under `vendor/seed-vc/checkpoints`; Demucs uses `~/.cache/torch/hub/checkpoints` unless `TORCH_HOME` overrides it. Removing caches requires downloading models again. Audio, caches, and job directories are excluded from Git.

## Troubleshooting

- **Model loading/download appears stuck:** inspect `process.log`. `python model_cache.py` resumes incomplete Demucs downloads and checks the upstream SHA-256 prefix. Use offline mode only after all required files are cached. Do not start another conversion concurrently.
- **MPS or memory error:** shorten the clip or use `--device cpu`; CPU inference is slower.
- **Silent/too-short reference:** provide continuous, clear audio of at least 3 seconds.
- **Accompaniment too short:** align both tracks and provide sufficient duration, or use song mode.
- **Long-running job:** individual model stages time out after 30 minutes; logs remain available.
- **Windows:** native Windows is not supported by this alpha's installer/process lock. WSL2 is an unverified option.

## Development and license

```bash
python -m pip install pytest
python -m pytest -q
```

Lightweight tests cover FFmpeg mixing/alignment, truncation, invalid/silent inputs, persisted errors, and concurrency protection. CI does not download large models or establish perceptual audio quality.

Code is GPL-3.0. Engines, weights, FFmpeg, and dependencies retain their own licenses; see [THIRD_PARTY.md](THIRD_PARTY.md). The small sample recordings and AI derivative retain CC BY 4.0; see examples/README.md. Model weights are not bundled. Use your own or authorized voices and songs, and label published output as AI-generated.

## Included samples

See [sample data and attribution](examples/README.md) for ready-to-use inputs, offline previews, licenses, and limitations. No API credentials are included.
