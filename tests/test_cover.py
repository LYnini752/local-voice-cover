import pytest
from cover import seed_command


def test_command_preserves_paths_and_singing_mode():
    args = seed_command("python", "/tmp/a $(no).wav", "/tmp/ref.wav", "/tmp/out", 30, -2)
    assert args[args.index("--source")+1] == "/tmp/a $(no).wav"
    assert args[args.index("--f0-condition")+1] == "True"
    assert args[args.index("--semi-tone-shift")+1] == "-2"


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        seed_command("python", "a", "b", "c", 1000, 0)

import json
from pathlib import Path
import numpy as np
import soundfile as sf
import cover


@pytest.fixture
def audio_files(tmp_path):
    rate = 44100
    t = np.arange(rate * 4) / rate
    reference = tmp_path / 'reference with spaces.wav'
    source = tmp_path / 'source $(literal).wav'
    sf.write(reference, .2 * np.sin(2 * np.pi * 220 * t), rate)
    sf.write(source, .1 * np.sin(2 * np.pi * 440 * t), rate)
    return reference, source


@pytest.mark.parametrize('steps,shift,gain', [(float('nan'), 0, 1), (20.5, 0, 1), (30, 1.5, 1), (30, 0, float('nan')), (30, 0, 3)])
def test_unbounded_and_fractional_settings_rejected(steps, shift, gain):
    with pytest.raises(ValueError):
        cover.validate_options(steps, shift, gain, 1, 'vocals', 'cpu')


def test_silence_and_nonfinite_audio_rejected(tmp_path):
    p = tmp_path / 'bad.wav'
    for values in [np.zeros(44100), np.full(44100, np.nan)]:
        sf.write(p, values, 44100, subtype='FLOAT')
        with pytest.raises(ValueError):
            cover.audio_stats(p)


def test_ffmpeg_mix_preserves_length_and_limits_peaks(audio_files, tmp_path):
    reference, source = audio_files
    data, rate = sf.read(reference)
    sf.write(reference, data * 4, rate)
    output = tmp_path / 'mix.wav'
    cover.mix_audio(cover.ffmpeg_path(), reference, source, output, 2, 2, tmp_path / 'log.txt')
    data, rate = sf.read(output)
    assert len(data) == 4 * rate  # Includes the limiter's compensated latency.
    assert 0.1 < np.max(np.abs(data)) <= 0.951


def test_ffmpeg_normalization_truncates_and_converts_channels(audio_files, tmp_path):
    output = tmp_path / 'normalized.wav'
    stats = cover.normalize(cover.ffmpeg_path(), audio_files[0], output, 1, 2, tmp_path / 'log.txt')
    assert stats['seconds'] == 1
    assert stats['channels'] == 2
    assert stats['sample_rate'] == 44100


def test_lock_prevents_overlapping_model_jobs(monkeypatch, tmp_path):
    monkeypatch.setattr(cover, 'JOBS', tmp_path / 'jobs')
    with cover.inference_lock():
        with pytest.raises(RuntimeError, match='已有转换任务'):
            with cover.inference_lock():
                pytest.fail('Nested process lock should not succeed')
    with cover.inference_lock():
        pass


def fake_model(monkeypatch, tmp_path, failure=False):
    monkeypatch.setattr(cover, 'JOBS', tmp_path / 'jobs')
    monkeypatch.setattr(cover, 'engine_paths', lambda: (tmp_path, Path('/unused/python')))
    real_execute = cover.execute

    def execute(args, cwd, log, timeout=1800):
        if str(args[0]) == '/unused/python':
            if failure:
                raise RuntimeError('Simulated engine failure')
            out = Path(args[args.index('--output') + 1])
            source = Path(args[args.index('--source') + 1])
            # A test stub validates orchestration, never quality or voice conversion.
            out.joinpath('test.wav').write_bytes(source.read_bytes())
        else:
            real_execute(args, cwd, log, timeout)
    monkeypatch.setattr(cover, 'execute', execute)


def test_pipeline_persists_complete_report_and_allows_muted_output(monkeypatch, tmp_path, audio_files):
    fake_model(monkeypatch, tmp_path)
    vocal, result, metadata = cover.convert(*audio_files, voice_gain=0)
    assert Path(vocal).is_file() and Path(result).is_file()
    report = json.loads(Path(metadata).read_text())
    assert report['status'] == 'complete'
    assert report['output']['peak'] == 0
    assert report['synthetic_audio'] is True
    assert len(report['output_sha256']) == 64


def test_pipeline_failure_is_recorded_without_success_audio(monkeypatch, tmp_path, audio_files):
    fake_model(monkeypatch, tmp_path, failure=True)
    with pytest.raises(RuntimeError, match='Simulated engine failure'):
        cover.convert(*audio_files)
    reports = list((tmp_path / 'jobs').glob('*/metadata.json'))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())['status'] == 'failed'
    assert not list((tmp_path / 'jobs').glob('*/result.wav'))


def test_ambiguous_song_plus_accompaniment_is_rejected(monkeypatch, tmp_path, audio_files):
    monkeypatch.setattr(cover, 'JOBS', tmp_path / 'jobs')
    with pytest.raises(ValueError, match='整曲模式'):
        cover.convert(*audio_files, accompaniment=audio_files[0], mode='song')


def test_stereo_accompaniment_is_not_downmixed(audio_files, tmp_path):
    rate=44100
    t=np.arange(2*rate)/rate
    music=tmp_path/'stereo.wav'
    sf.write(music,np.column_stack([.2*np.sin(2*np.pi*220*t), .2*np.sin(2*np.pi*880*t)]),rate)
    result=tmp_path/'stereo-result.wav'
    cover.mix_audio(cover.ffmpeg_path(),audio_files[0],music,result,0,1,tmp_path/'mix.log')
    samples,_=sf.read(result,always_2d=True)
    assert samples.shape[1]==2
    assert abs(np.corrcoef(samples[:rate].T)[0,1])<.01
