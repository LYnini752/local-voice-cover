"""Resumable download of the official Demucs checkpoint, with upstream hash check."""
import hashlib
import os
from pathlib import Path
import re
import time
import urllib.request

DEMUCS_URL = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th"
DEMUCS_HASH_PREFIX = "8726e21a"  # Published in the upstream checkpoint filename.


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def demucs_checkpoint():
    cache = Path(os.getenv('TORCH_HOME', str(Path(os.getenv('XDG_CACHE_HOME', '~/.cache')).expanduser() / 'torch'))).expanduser()
    return cache / 'hub' / 'checkpoints' / DEMUCS_URL.rsplit('/', 1)[1]


def download_verified(url, destination, hash_prefix, timeout=30, total_timeout=1800):
    """Keep partial bytes on failure; only a verified file becomes the final cache entry."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.partial')
    if destination.exists():
        if not digest(destination).startswith(hash_prefix):
            raise RuntimeError(f'模型缓存校验失败，请移走损坏文件后重试：{destination}')
        return destination
    if partial.exists() and digest(partial).startswith(hash_prefix):
        os.replace(partial, destination)
        return destination
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {'Range': f'bytes={offset}-'} if offset else {}
    started = time.monotonic()
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
        if response.status == 206:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
            if not match or int(match.group(1)) != offset:
                raise RuntimeError('模型下载服务器返回了不匹配的续传范围；已保留已有数据')
            mode = 'ab'
        elif response.status == 200:
            mode = 'wb'  # The server ignored Range, so restart instead of appending duplicate bytes.
        else:
            raise RuntimeError(f'模型下载返回意外状态：{response.status}')
        with partial.open(mode) as stream:
            while True:
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                stream.flush()
                if time.monotonic() - started > total_timeout:
                    raise RuntimeError('模型下载超时，已保留部分文件；再次运行可尝试续传')
    if not digest(partial).startswith(hash_prefix):
        raise RuntimeError(f'模型下载未通过 SHA-256 前缀校验，未启用该文件：{partial}')
    os.replace(partial, destination)
    return destination


def ensure_demucs():
    import fcntl
    target = demucs_checkpoint()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_name(target.name + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return download_verified(DEMUCS_URL, target, DEMUCS_HASH_PREFIX)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == '__main__':
    print('下载 / 检查官方 Demucs 模型，失败时保留断点。', flush=True)
    print(ensure_demucs(), flush=True)
