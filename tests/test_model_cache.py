import hashlib
import io
from pathlib import Path
import pytest
import model_cache


class Response(io.BytesIO):
    def __init__(self, payload, status, headers):
        super().__init__(payload)
        self.status = status
        self.headers = headers


def test_resumes_from_exact_offset_and_verifies(monkeypatch, tmp_path):
    target = tmp_path / 'model.th'
    target.with_name('model.th.partial').write_bytes(b'first')
    data = b'firstsecond'
    def open_request(request, timeout):
        assert request.headers['Range'] == 'bytes=5-'
        return Response(b'second', 206, {'Content-Range': 'bytes 5-10/11'})
    monkeypatch.setattr(model_cache.urllib.request, 'urlopen', open_request)
    model_cache.download_verified('https://example.test/model', target, hashlib.sha256(data).hexdigest())
    assert target.read_bytes() == data
    assert not target.with_name('model.th.partial').exists()


def test_range_ignored_restarts_instead_of_appending(monkeypatch, tmp_path):
    target = tmp_path / 'model.th'
    target.with_name('model.th.partial').write_bytes(b'prefix')
    data = b'complete'
    monkeypatch.setattr(model_cache.urllib.request, 'urlopen', lambda *a, **k: Response(data, 200, {}))
    model_cache.download_verified('https://example.test/model', target, hashlib.sha256(data).hexdigest())
    assert target.read_bytes() == data


def test_wrong_range_does_not_damage_partial(monkeypatch, tmp_path):
    target = tmp_path / 'model.th'
    partial = target.with_name('model.th.partial')
    partial.write_bytes(b'first')
    monkeypatch.setattr(model_cache.urllib.request, 'urlopen', lambda *a, **k: Response(b'bad', 206, {'Content-Range': 'bytes 2-4/5'}))
    with pytest.raises(RuntimeError, match='续传范围'):
        model_cache.download_verified('https://example.test/model', target, 'deadbeef')
    assert partial.read_bytes() == b'first'
    assert not target.exists()


def test_hash_failure_never_activates_weight_file(monkeypatch, tmp_path):
    target = tmp_path / 'model.th'
    monkeypatch.setattr(model_cache.urllib.request, 'urlopen', lambda *a, **k: Response(b'corrupt', 200, {}))
    with pytest.raises(RuntimeError, match='SHA-256'):
        model_cache.download_verified('https://example.test/model', target, 'deadbeef')
    assert not target.exists()


def test_verified_cached_file_does_not_access_network(monkeypatch, tmp_path):
    target = tmp_path / 'model.th'
    target.write_bytes(b'complete')
    def fail(*args, **kwargs):
        pytest.fail('Network should not be accessed')
    monkeypatch.setattr(model_cache.urllib.request, 'urlopen', fail)
    model_cache.download_verified('https://example.test/model', target, hashlib.sha256(b'complete').hexdigest())
