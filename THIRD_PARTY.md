# Third-party components

- [Seed-VC](https://github.com/Plachtaa/seed-vc): GPL-3.0; pinned at `51383efd921027683c89e5348211d93ff12ac2a8`. Cloned by the installer, not vendored in this repository. The MPS adapter converts RMVPE's NumPy F0 result to float32 and moves BigVGAN vocoding to CPU without editing the upstream checkout.
- [Demucs](https://github.com/facebookresearch/demucs): MIT; package 4.0.1, `htdemucs` separator. Installed separately.
- [BigVGAN](https://github.com/NVIDIA/BigVGAN): MIT; used by the upstream engine.
- UI: [Gradio](https://github.com/gradio-app/gradio), Apache-2.0.
- FFmpeg is supplied by [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg), or an existing system installation. Binary licensing depends on the build; inspect its `-L` output if redistributing it. This repository does not redistribute binaries.

Model weights download from their publishers on first use: `Plachta/Seed-VC`, `lj1995/VoiceConversionWebUI`, `funasr/campplus`, `nvidia/bigvgan_v2_44khz_128band_512x`, `openai/whisper-small`, and the official Demucs weight host. Each weight/model card has its own terms; this repository's GPL-3.0 license does not relicense those weights or grant rights to voices/music.

No reference voices, songs, third-party sample audio or model weights are published in this repository. Local developer validation may use the upstream checkout's examples without redistributing them.
