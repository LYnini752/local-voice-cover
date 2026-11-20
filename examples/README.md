# 免费清唱示例 / Free singing samples

点“加载免费清唱示例”可填入两段 8 秒音频，自动选纯人声模式和 10 步，再点开始转换。展开“直接试听预计算结果”可立即播放已经生成的效果，不需要先下载模型；参考音频、目标清唱都可在输入播放器里听。

## 来源与许可 / Attribution

Dataset: **vocadito v3**, Rachel M. Bittner, Katherine Pasalo, Juan José Bosch, Gabriel Meseguer Brocal, David Rubinstein (2021). DOI: [10.5281/zenodo.5578807](https://zenodo.org/records/5578807). Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

| 本地文件 | 原文件 | 变更 |
|---|---|---|
| reference.flac | Audio/vocadito_11.wav，匿名歌手 S8，英语 | 取前 8 秒，无损 FLAC 编码 |
| singing.flac | Audio/vocadito_35.wav，匿名歌手 S28，普通话 | 取前 8 秒，无损 FLAC 编码 |
| converted-ai.flac | 上述两个文件的派生输出 | Seed-VC 换音色：10 步、CPU、零移调、无伴奏；AI 合成，非真人原始演唱 |

源文件、输入和派生结果哈希见 manifest.json。数据作者及歌手不为本项目背书。素材保留 CC BY 4.0，仓库 GPL 许可不替代素材许可；再分享时保留署名、许可证链接及改动说明，并标明 AI 合成。

vocadito 论文说明志愿者选用原创或公版歌曲，并同意匿名公开录音：[技术说明](https://archives.ismir.net/ismir2021/latebreaking/000044.pdf)。这里只提供两段小样例，不打包整个数据集。真实音乐录制条件和音域会影响效果；这不是声纹保真或歌唱质量基准。

## English

Click the sample button to fill two 8-second clips, select vocals-only mode and 10 steps. Open the precomputed-result accordion to listen immediately, or run a fresh conversion. These two anonymized singers come from the credited CC BY 4.0 vocadito dataset; inputs are cropped to 8 seconds and losslessly encoded as FLAC. The converted sample is AI-generated and is not a genuine performance by the reference singer. Preserve attribution, license, and modification disclosure when sharing. No endorsement is implied.
