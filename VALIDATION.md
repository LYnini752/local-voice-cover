# Validation status

As of 2026-11-19. This document reports observed checks, not planned capabilities.

## Environment

- macOS, Apple Silicon, 16 GB unified memory.
- UI Python 3.13.3; model Python 3.10.22.
- PyTorch/torchaudio 2.5.1, Seed-VC commit `51383efd921027683c89e5348211d93ff12ac2a8`.
- First-use weights downloaded from upstream publishers; not committed.

## Verified so far

- Upstream inference import, dependency check and installation doctor.
- A six-second **speech** input converted in singing (`f0_condition=True`) mode with 10 diffusion steps. Output: 5.991 seconds, 44.1 kHz mono, nonzero signal. Entire warm-cache pipeline took 60.16 seconds on this machine, including model loading. This is a pipeline smoke test, not a benchmark or a singing-quality endorsement.
- Found and addressed two MPS failures: float64 RMVPE F0 arrays, and BigVGAN grouped transposed convolution. Adapter converts F0 to float32 and performs final vocoding on CPU, leaving upstream source unchanged.
- 20 lightweight tests passed: malformed options, silent/nonfinite input, actual FFmpeg normalization/mixing, exclusive job locking, successful/failed job bookkeeping with a stub model, and resumable download correctness/hash checking.
- Gradio application constructs successfully; localhost config and a real client upload/API request passed. A concurrent model request was correctly rejected. Desktop browser layout inspected. Inputs, mode selection, advanced options and output sections render correctly. A Markdown emphasis spacing issue found during inspection was corrected.

- Audited all 267 learned tensors in the conversion checkpoint: no missing learned tensors or shape mismatches after upstream DDP prefix normalization. Ten unused legacy checkpoint tensors explain the upstream warning.
- Speech pitch sanity check on jointly voiced frames: median absolute difference 24.47 cents, 93.6% within 100 cents. This is one six-second speech case, not a singing benchmark.

- Ten-second real singing excerpt converted with 30 steps: 9.996-second output, nonzero signal, 44.1 kHz, peak limited to 0.950012 after PCM quantization. End-to-end time: 85.92 seconds with cached weights and Hugging Face offline mode enabled. The source is an upstream developer example used locally only.

- Pure CPU conversion completed: six-second speech input to 5.991-second WAV, 52.88 seconds end-to-end on this Mac. Hugging Face offline mode succeeded.
- The ten-second singing sanity check had 904 jointly voiced frames: median absolute F0 difference 11.87 cents, 94.9% within 100 cents. One source/reference pair does not establish general singing or voice-similarity quality.

- A 65-second repeated singing fixture was correctly truncated to 60 seconds and converted across multiple chunks: output 59.989 seconds, 309.94 seconds end-to-end. This tests duration/chunking, not musical quality.
- Full-song mode completed on ten seconds of upstream singing mixed locally with synthetic accompaniment: Demucs separation → Seed-VC 30 steps → remix, 89.12 seconds. Inspection exposed a stereo downmix bug; the mixer now explicitly preserves stereo, with a separate-channel regression test and real-stem remix check.
- A fresh isolated UI installation passed all 20 tests, app construction and CLI doctor. The first run of the expanded suite after the stereo fix uses the same clean environment.

- Fresh isolated UI HTTP/API run completed the full song pipeline with 30 steps: 9.996 seconds, stereo, 95.98 seconds end-to-end. Missing-input validation returned the expected error.
- Compressed MP3 source, separate FLAC accompaniment, 30-second reference (clipped to 25 seconds), +4-semitone option and CPU processing completed together: 5.991-second stereo output, 161.81 seconds. This confirms the option path, not an independent pitch-shift quality benchmark.
- On one anonymous speech pair, the conditioning encoder cosine similarity to the reference increased from 0.238 (source) to 0.680 (converted). Because this is the same encoder used to condition conversion, it is only a sanity check, not independent speaker verification.

## In progress / not yet verified

- Broader full-song separation/remix quality on real music recordings.
- Broader singing quality review and F0 preservation across multiple references.
- CUDA/Linux.
- Human listening and voice-similarity review with user-approved samples.

Upstream anonymous speech examples were used locally for developer testing. They and derived outputs are not published with this repository. Synthetic music mixed with speech is useful for plumbing checks, but does not replace testing real singing.
