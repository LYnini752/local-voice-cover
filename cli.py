"""Headless entry point for conversion or installation checks."""
import argparse
import json
from pathlib import Path
from cover import convert, engine_paths, ffmpeg_path


def main():
    parser = argparse.ArgumentParser(description="Local singing voice conversion")
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--accompaniment", type=Path)
    parser.add_argument("--mode", choices=["song", "vocals"], default="song")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--shift", type=int, default=0)
    args = parser.parse_args()
    if args.doctor:
        try:
            engine, python = engine_paths()
            print(json.dumps({"engine": str(engine), "python": str(python), "ffmpeg": ffmpeg_path()}, indent=2))
        except ValueError as exc:
            parser.exit(1, str(exc) + "\n")
        return
    if not args.reference or not args.source:
        parser.error("--reference and --source are required")
    try:
        result = convert(args.reference, args.source, args.accompaniment, args.steps, args.shift,
                         mode=args.mode, device=args.device,
                         progress=lambda _, desc: print(desc, flush=True))
        print(json.dumps(dict(zip(["vocal", "result", "metadata"], result)), ensure_ascii=False, indent=2))
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
