"""Launch the PetVision Gradio demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
from PIL import Image

from petvision.inference import PetVisionInference


def _rows(values: list[tuple[str, float]]) -> list[list[object]]:
    return [[name, round(probability, 6)] for name, probability in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segmentation-checkpoint",
        type=Path,
        default=Path("artifacts/segmentation/best.pt"),
    )
    parser.add_argument(
        "--raw-checkpoint",
        type=Path,
        default=Path("artifacts/classification/raw/best.pt"),
    )
    parser.add_argument(
        "--guided-checkpoint",
        type=Path,
        default=Path("artifacts/classification/guided/best.pt"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    service = PetVisionInference.from_checkpoints(
        args.segmentation_checkpoint,
        args.raw_checkpoint,
        args.guided_checkpoint,
        args.device,
    )

    def predict(image: Image.Image):
        if image is None:
            return None, None, None, [], [], "请先上传图片。"
        result = service.predict(image)
        mask = Image.fromarray((result.mask * 255).astype("uint8"), mode="L")
        raw_top1 = result.raw_top5[0]
        guided_top1 = result.guided_top5[0]
        summary = (
            f"直接分类：**{raw_top1[0]}**（{raw_top1[1]:.2%}）\n\n"
            f"分割辅助分类：**{guided_top1[0]}**（{guided_top1[1]:.2%}）"
        )
        return (
            result.overlay,
            mask,
            result.foreground_image,
            _rows(result.raw_top5),
            _rows(result.guided_top5),
            summary,
        )

    with gr.Blocks(title="PetVision") as demo:
        gr.Markdown(
            "# PetVision\n"
            "自然场景宠物分割与分类演示。系统会真实调用分割模型和两个分类模型。"
        )
        with gr.Row():
            input_image = gr.Image(type="pil", label="输入图像")
            with gr.Column():
                summary = gr.Markdown(label="预测摘要")
                run_button = gr.Button("开始推理", variant="primary")
        with gr.Row():
            overlay = gr.Image(label="分割叠加结果")
            mask = gr.Image(label="宠物前景 mask")
            foreground = gr.Image(label="提取的宠物主体")
        with gr.Row():
            raw_table = gr.Dataframe(
                headers=["品种", "概率"],
                datatype=["str", "number"],
                label="直接分类 Top-5",
            )
            guided_table = gr.Dataframe(
                headers=["品种", "概率"],
                datatype=["str", "number"],
                label="分割辅助分类 Top-5",
            )
        run_button.click(
            predict,
            inputs=input_image,
            outputs=[overlay, mask, foreground, raw_table, guided_table, summary],
        )
        input_image.change(
            predict,
            inputs=input_image,
            outputs=[overlay, mask, foreground, raw_table, guided_table, summary],
        )

    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main()
