import os
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr
from cover import convert


def run(reference, source, music, steps, shift, vg, mg, mode, device, progress=gr.Progress()):
    try:
        return convert(reference, source, music, steps, shift, vg, mg, mode, device, progress)
    except Exception as exc:
        raise gr.Error(str(exc)) from None


def build_app():
    with gr.Blocks(title="Local Voice Cover", theme=gr.themes.Soft(), analytics_enabled=False) as demo:
        gr.Markdown("# 本地 AI 翻唱\n上传参考声音和目标歌曲，生成换音色后的音频。模型在本机运行，首次使用需要联网下载。")
        gr.Markdown("**短片段预览版：**参考取前 25 秒，歌曲取前 60 秒。建议先试 10–20 秒片段。请使用你有权使用的声音与歌曲，分享时标明 AI 合成。")
        with gr.Row():
            reference = gr.Audio(type="filepath", label="① 参考声音（至少 3 秒，尽量干净）")
            source = gr.Audio(type="filepath", label="② 目标歌曲 / 目标人声")
        mode = gr.Radio([("完整歌曲：自动分离并保留伴奏", "song"), ("已分离的纯人声", "vocals")], value="song", label="目标音频类型")
        with gr.Accordion("调整与高级设置", open=False):
            music = gr.Audio(type="filepath", label="纯人声模式可选：同一时间点开始的伴奏")
            with gr.Row():
                steps = gr.Slider(10, 100, value=30, step=1, label="转换步数（越高越慢）")
                shift = gr.Slider(-12, 12, value=0, step=1, label="人声移调（半音，不改变伴奏）")
            vg = gr.Slider(0, 2, value=1, label="人声音量")
            mg = gr.Slider(0, 2, value=0.8, label="伴奏音量")
            device = gr.Dropdown(["auto", "cpu", "mps", "cuda"], value="auto", label="转换设备（内存不足可改 CPU）")
        button = gr.Button("开始转换", variant="primary")
        with gr.Row():
            vocal = gr.Audio(label="转换后人声")
            mixed = gr.Audio(label="最终音频 · AI 合成")
        metadata = gr.File(label="下载转换记录")
        gr.Markdown("输入与结果保存在本机 `.local/jobs/`，不会自动删除；清理方法见 README。网页仅监听本机。")
        button.click(run, [reference, source, music, steps, shift, vg, mg, mode, device], [vocal, mixed, metadata], concurrency_limit=1)
    return demo


if __name__ == "__main__":
    build_app().queue(max_size=4).launch(server_name="127.0.0.1", server_port=int(os.getenv("PORT", "7862")), share=False, max_file_size="500mb")
