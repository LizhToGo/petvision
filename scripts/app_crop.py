"""Run the trained crop classifier locally, with optional raw-model comparison."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

from petvision.inference import PetVisionCropInference


def build_demo(
    service: PetVisionCropInference, examples: list[Path] | None = None
) -> gr.Blocks:
    """Create the upload UI around a single loaded inference service."""

    def predict(image: Image.Image | None):
        if image is None:
            return None, None, None, {}, {}, "请先上传图片。"
        started = time.perf_counter()
        result = service.predict(image)
        elapsed = time.perf_counter() - started
        localized = result.overlay.copy()
        left, top, right, bottom = result.crop_box
        width, height = localized.size
        ImageDraw.Draw(localized).rectangle(
            (left * (width - 1), top * (height - 1), right * (width - 1), bottom * (height - 1)),
            outline="red",
            width=2,
        )
        fraction = (right - left) * (bottom - top)
        status = (
            f"推理耗时：{elapsed:.3f} 秒；设备：{service.device}。\n\n"
            f"裁剪框占原图面积：{fraction:.1%}；"
            f"前景阈值：{service.threshold:g}；边距：{service.margin:g}。\n\n"
            "显示的是 softmax 模型得分，不是经过校准的正确概率。"
        )
        if result.crop_fallback:
            status += "\n\n未检测到前景，本次按训练约定使用整图分类。"
        return (
            localized,
            Image.fromarray(result.mask * 255),
            result.crop_image,
            dict(result.raw_top5 or []),
            dict(result.crop_top5),
            status,
        )

    with gr.Blocks(title="PetVision · 本地模型演示") as demo:
        gr.Markdown(
            "# PetVision · 本地模型演示\n"
            "上传图片后，真实运行 U-Net 定位和 ResNet-50 主体裁剪分类。"
            "红框是实际裁剪范围，绿色是预测前景；裁剪参数从训练 checkpoint 读取。\n\n"
            "**适用范围：37 个已训练的宠物品种，单个主要目标。**"
            "模型没有无宠物或未知品种拒识功能，多宠物、非宠物图片也可能得到高分。"
        )
        with gr.Row():
            input_image = gr.Image(type="pil", sources=["upload"], label="上传你的宠物照片", elem_id="pet-input")
            with gr.Column():
                status = gr.Markdown("选择图片后自动推理，也可以点击按钮重新运行。", elem_id="prediction-status")
                run_button = gr.Button("开始推理", variant="primary", elem_id="predict-button")
        with gr.Row():
            overlay = gr.Image(label="预测前景与实际裁剪框", interactive=False, height=300)
            mask = gr.Image(label="预测前景 mask", interactive=False, height=300)
            local_image = gr.Image(label="从原始分辨率裁剪的主体", interactive=False, height=300)
        with gr.Row():
            raw_scores = gr.Label(num_top_classes=5, label="原图模型 Top-5", visible=service.raw_classifier is not None)
            crop_scores = gr.Label(num_top_classes=5, label="主体裁剪模型 Top-5")
        outputs = [overlay, mask, local_image, raw_scores, crop_scores, status]
        run_button.click(predict, inputs=input_image, outputs=outputs, api_name="predict")
        input_image.upload(predict, inputs=input_image, outputs=outputs, api_name=False)
        input_image.clear(lambda: predict(None), outputs=outputs, api_name=False)
        if examples:
            gr.Examples(examples=[[str(path.resolve())] for path in examples], inputs=input_image)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--crop-checkpoint", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--examples", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    service = PetVisionCropInference.from_checkpoints(
        args.segmentation_checkpoint,
        args.crop_checkpoint,
        device=args.device,
        raw_checkpoint=args.raw_checkpoint,
    )
    print(
        f"Loaded crop classifier on {service.device}; "
        f"segmentation_size={service.segmentation_size} classification_size={service.image_size} "
        f"threshold={service.threshold} margin={service.margin}",
        flush=True,
    )
    demo = build_demo(service, args.examples)
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=args.open_browser)


if __name__ == "__main__":
    main()
