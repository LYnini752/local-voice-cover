"""Install the pinned upstream engine into this project, without modifying system Python."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REVISION = "51383efd921027683c89e5348211d93ff12ac2a8"
REPOSITORY = "https://github.com/Plachtaa/seed-vc.git"


def run(args):
    subprocess.run([str(x) for x in args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check installation without downloading")
    args = parser.parse_args()
    engine = ROOT / "vendor" / "seed-vc"
    python = ROOT / ".venv-engine" / "bin" / "python"
    if args.check:
        if not (engine / "inference.py").is_file() or not python.is_file():
            raise SystemExit("Engine not installed. Run without --check first.")
        run([python, ROOT / "seed_runner.py", "--help"])
        actual = subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"], text=True).strip()
        if actual != REVISION:
            raise SystemExit("Upstream revision differs from tested pin")
        run([python, "-c", "import torch, torchaudio, librosa, demucs; print('Engine dependencies OK; MPS:', torch.backends.mps.is_available())"])
        return
    if not shutil.which("git"):
        raise SystemExit("Install Git first.")
    # uv is an explicit installer dependency; no curl-pipe-shell or global installs.
    uv = shutil.which("uv")
    if uv is None:
        candidate = Path(sys.executable).parent / "uv"
        uv = str(candidate) if candidate.is_file() else None
    if not uv:
        raise SystemExit("Run: python -m pip install uv==0.13.9 (inside the UI virtual environment)")
    if engine.exists():
        if not (engine / ".git").exists():
            raise SystemExit(f"Refusing to replace existing non-Git directory: {engine}")
        actual = subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(engine), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        if actual != REVISION or dirty:
            raise SystemExit("Existing engine differs from pin or has edits; choose a fresh checkout manually. No files were reset.")
    else:
        engine.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", REPOSITORY, engine])
        run(["git", "-C", engine, "checkout", "--detach", REVISION])
    if not python.exists():
        run([uv, "venv", "--python", "3.10", ROOT / ".venv-engine"])
    version = json.loads(subprocess.check_output([python, "-c", "import sys,json; print(json.dumps(list(sys.version_info[:2])))"], text=True))
    if version != [3, 10]:
        raise SystemExit("Existing .venv-engine must use Python 3.10. Move it aside and rerun; no environment was overwritten.")
    run([uv, "pip", "install", "--python", python, "-r", ROOT / "requirements-engine.txt"])
    print("Installed. Models download on first conversion; no audio is uploaded by this pipeline.")


if __name__ == "__main__":
    main()
