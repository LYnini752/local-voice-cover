"""Thin adapter for a pinned, unmodified upstream Seed-VC checkout."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--semi-tone-shift", type=int, default=0)
    parser.add_argument("--f0-condition", choices=["True"], default="True")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    engine = args.engine_dir.resolve()
    sys.path.insert(0, str(engine))
    os.chdir(engine)
    import torch
    import inference
    if args.device != "auto":
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS is unavailable")
        inference.device = torch.device(args.device)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    # Upstream RMVPE returns NumPy float64. MPS cannot transfer float64 tensors.
    # Restrict this compatibility adapter to the F0 result; never change torch globally.
    if inference.device.type == "mps":
        import numpy as np
        from modules.rmvpe import RMVPE
        original_infer = RMVPE.infer_from_audio

        def infer_float32(self, *values, **options):
            return np.asarray(original_infer(self, *values, **options), dtype=np.float32)

        RMVPE.infer_from_audio = infer_float32
        # PyTorch 2.5 MPS fails on BigVGAN's grouped transposed convolution.
        # Keep diffusion on MPS and run the final waveform decoder on CPU.
        original_load = inference.load_models

        def load_with_cpu_vocoder(options):
            components = list(original_load(options))
            vocoder = components[3].to("cpu")
            components[3] = lambda mel: vocoder(mel.to("cpu"))
            print("BigVGAN decoder: CPU (MPS convolution compatibility)", flush=True)
            return tuple(components)

        inference.load_models = load_with_cpu_vocoder
    print(f"Seed-VC device: {inference.device}", flush=True)
    args.f0_condition = True
    args.auto_f0_adjust = False
    args.fp16 = False
    args.length_adjust = 1.0
    args.inference_cfg_rate = 0.7
    args.checkpoint = None
    args.config = None
    inference.main(args)


if __name__ == "__main__":
    main()
