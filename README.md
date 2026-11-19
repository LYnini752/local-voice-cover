# Local Voice Cover · 本地 AI 翻唱

两段音频输入：**参考声音 + 想唱的歌曲**。在本机提取音色条件、分离歌曲人声、进行歌声转换，再把转换后人声混回伴奏，导出 WAV。无需为每个声音单独训练模型。

这是基于 [Seed-VC](https://github.com/Plachtaa/seed-vc) 与 [Demucs](https://github.com/facebookresearch/demucs) 的短片段应用，不是新训练的基础模型。目标是让已有开源模型容易在本地使用和复现。

## 首版范围

- 完整歌曲模式：自动人声分离 → 音色转换 → 保留伴奏。
- 纯人声模式：直接转换，可另传从同一时刻开始的伴奏。
- 参考取前 **25 秒**，歌曲取前 **60 秒**。参考最短 3 秒，建议 6–25 秒清晰独唱或说话。
- 步数、半音移调、人声和伴奏音量；下载音频与 JSON 转换记录。
- 本地网页及命令行；进度、失败日志、单任务锁。
- Apple Silicon 的 MPS 兼容适配：音高转 float32；扩散模型在 MPS，BigVGAN 解码在 CPU。

**Alpha，非实时。** 声线相似度、咬字和音乐质量取决于输入。参考有背景乐、多人重叠、高音跨度很大时，效果可能下降。移调只改变转换人声，不改变伴奏调性。并非一键处理无限长歌曲。

## 快速开始（macOS / Linux）

需要 Git、Python 3.11–3.13；模型环境由安装器单独创建为 Python 3.10。首次下载依赖和权重需联网，并预留约 10 GB 磁盘空间。推荐先在 16 GB 内存电脑上尝试 10 秒片段；这不是对所有 16 GB 设备的性能保证。

下载或克隆本仓库后，在仓库目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt uv==0.13.9
python scripts/setup_engine.py
python scripts/setup_engine.py --check
python cli.py --doctor
python model_cache.py  # 可选：提前下载人声分离模型，支持断点续传
python app.py
```

打开 <http://127.0.0.1:7862>。依次上传两段音频、选择目标音频类型、点击“开始转换”。首次转换会从模型发布方下载权重；以后使用本地缓存。

安装器固定 Seed-VC 提交和主要模型依赖，保留现有的修改，不会自动重置已有引擎目录。`requirements-engine.txt` 是独立模型环境依赖，**不要与 UI 环境混装**。`requirements-engine.lock.txt` 记录此次 Mac 实测环境的完整版本，跨平台安装仍以主要依赖清单为入口。

## 无界面运行

```bash
python cli.py --reference '/path/to/reference.wav' \
  --source '/path/to/song.mp3' --mode song --steps 30

python cli.py --reference '/path/to/reference.wav' \
  --source '/path/to/vocals.wav' --mode vocals --device cpu
```

使用 `--accompaniment '/path/to/music.wav'` 可在纯人声模式混入伴奏。路径中包含空格时加引号。`--shift -2` 表示人声降低两个半音。

可选环境变量：

| 变量 | 默认值 | 用途 |
|---|---|---|
| `SEED_VC_DIR` | 仓库下 `vendor/seed-vc` | 已有的兼容引擎路径 |
| `SEED_VC_PYTHON` | 仓库下 `.venv-engine/bin/python` | 引擎 Python 的绝对路径 |
| `FFMPEG_BINARY` | 系统 FFmpeg，或 imageio 提供的二进制 | 指定音频工具 |
| `PORT` | `7862` | 本地网页端口 |
| `HF_HUB_OFFLINE` | 未设置 | 缓存齐全后设为 `1`，离线加载 Hugging Face 模型 |

`auto` 优先 CUDA，其次 Apple MPS，否则 CPU。分离器使用 CPU，避免与转换模型竞争显存。Linux/CUDA 和纯 CPU 的兼容程度见 [验证记录](VALIDATION.md)，不要把代码支持等同于每种硬件已实测。

## 输出、隐私与清理

每次任务写入 `.local/jobs/<任务编号>/`：

- `vocal.wav`：转换后人声。
- `result.wav`：最终音频（AI 合成）。
- `metadata.json`：处理状态、参数、时长、输出散列。
- `process.log`：失败原因和模型日志。
- 归一化输入、可选分离结果、中间音频。

本应用不把输入音频上传到推理服务，Gradio 统计上报已关闭；模型下载需要访问 GitHub、Hugging Face 和 Demucs 权重地址。网页仅绑定 `127.0.0.1`，不创建公共分享链接。Gradio 的临时上传文件也可能保留在系统临时目录。

任务不会自动清理。完成下载后，可以用文件管理器删除不需要的 `.local/jobs/<任务编号>` 文件夹；请勿删除运行中的任务。`vendor/seed-vc/checkpoints` 是转换模型缓存，Demucs 模型默认在 `~/.cache/torch/hub/checkpoints`（或 `TORCH_HOME` 指定目录）。删除后下次需重新下载。所有这些目录和音频都被 `.gitignore` 排除。

## 排错

- **首次停在模型加载**：查看 `process.log`；Demucs 下载失败会保留 `.partial` 文件，可运行 `python model_cache.py` 续传，下载完成会检查上游发布的 SHA-256 前缀。网络慢会重复下载检查。完整缓存后可用 `HF_HUB_OFFLINE=1` 重试。不要同时启动第二个任务。
- **MPS / 内存错误**：先缩短输入，使用 `--device cpu`；关闭其他占用内存的程序。CPU 较慢。
- **输入近似静音 / 参考不足 3 秒**：换成连续、清晰的声音片段。
- **伴奏短于人声**：裁剪成起点相同、长度足够的两轨，或使用完整歌曲模式。
- **任务超过 30 分钟**：单个模型阶段会超时停止，日志保留。转换不是实时服务。
- **Windows**：首版安装脚本和进程锁面向 macOS/Linux；Windows 原生未支持，可自行在 WSL2 尝试，尚未验证。

## 开发与验证

```bash
python -m pip install pytest
python -m pytest -q
```

轻量测试覆盖 FFmpeg 混音/对齐、截断、静音与无效参数、错误持久化和并发保护。CI 不下载大型模型，不代表模型质量已被验证。真实运行与已知限制见 [VALIDATION.md](VALIDATION.md)。

## 开源与使用

本仓库代码使用 GPL-3.0。引擎、权重、FFmpeg 和其他依赖各自的许可证见 [THIRD_PARTY.md](THIRD_PARTY.md)。本仓库不分发第三方歌曲、参考声音或模型权重。

请使用自己或已获授权的声音与歌曲；发布生成结果时标明 AI 合成。上传一段声音不会使你取得该声音或歌曲的其他使用权。
