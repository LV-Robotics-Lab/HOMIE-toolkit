"""
Batch-decompose repository videos into layered videos with Qwen Image Layered.

Outputs for each input video:
  - layer_XX_alpha.mov: per-layer video with alpha (ProRes 4444)
  - preview_grid.mp4: original frame plus all layers over a checkerboard
  - manifest.json: run metadata and resolved settings
  - prompts.jsonl: one prompt record per processed frame

Typical smoke test:
  .venv/bin/python examples/example_qwen_layered_videos.py \
      --scan-root data/xperience-10m-sample \
      --max-videos 1 \
      --max-frames 8 \
      --frame-stride 20

Full batch:
  .venv/bin/python examples/example_qwen_layered_videos.py \
      --scan-root data \
      --output-root outputs/qwen-layered \
      --frame-stride 1
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# Add package root to path
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils.caption_utils import load_caption_data_from_annotation_hdf5
from utils.video_utils import discover_video_files, get_video_metadata, iter_video_frames


@dataclass
class EpisodeCaptionContext:
    annotation_path: Path
    main_task: str
    frame_info_map: dict | None


class FFmpegRawVideoWriter:
    """Write raw RGB(A) frames into a video file through ffmpeg."""

    def __init__(self, output_path, width, height, fps, input_pix_fmt, codec_args, overwrite=False):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            raise RuntimeError("ffmpeg not found in PATH.")

        overwrite_flag = "-y" if overwrite else "-n"
        cmd = [
            ffmpeg_bin,
            overwrite_flag,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pix_fmt",
            input_pix_fmt,
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            *codec_args,
            str(self.output_path),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._closed = False

    def write(self, frame_array):
        if self._closed or self._proc.stdin is None:
            raise RuntimeError(f"Writer already closed: {self.output_path}")
        self._proc.stdin.write(np.asarray(frame_array, dtype=np.uint8).tobytes())

    def close(self, abort=False):
        if self._closed:
            return
        self._closed = True
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        if abort:
            self._proc.terminate()
            self._proc.wait(timeout=10)
            return
        stderr = b""
        if self._proc.stderr is not None:
            stderr = self._proc.stderr.read()
        return_code = self._proc.wait()
        if return_code != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed for {self.output_path}: {message or return_code}")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch layer repository videos with Qwen Image Layered.")
    parser.add_argument("--scan-root", type=str, default="data", help="Directory to recursively scan for videos.")
    parser.add_argument("--output-root", type=str, default="outputs/qwen-layered", help="Output directory for layered videos.")
    parser.add_argument("--include-glob", action="append", default=[], help="Optional relative-path glob filter, repeatable.")
    parser.add_argument("--max-videos", type=int, default=None, help="Only process the first N discovered videos.")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame index to process.")
    parser.add_argument("--end-frame", type=int, default=None, help="Stop before this frame index.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum processed frames per video after stride.")
    parser.add_argument("--layers", type=int, default=4, help="Number of output layers to request from the pipeline.")
    parser.add_argument("--resolution", type=int, choices=[640, 1024], default=640, help="Qwen Image Layered resolution bucket.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Diffusion denoising steps per frame.")
    parser.add_argument("--true-cfg-scale", type=float, default=4.0, help="Classifier-free guidance scale.")
    parser.add_argument("--negative-prompt", type=str, default=" ", help="Negative prompt. Keep a single space to enable CFG.")
    parser.add_argument("--cfg-normalize", action="store_true", help="Enable CFG normalization.")
    parser.add_argument("--use-en-prompt", action="store_true", default=True, help="Use automatic English captioning when prompt is empty.")
    parser.add_argument("--disable-use-en-prompt", action="store_false", dest="use_en_prompt", help="Disable automatic English captioning.")
    parser.add_argument("--prompt", type=str, default=None, help="Fixed prompt for every frame.")
    parser.add_argument(
        "--prompt-mode",
        type=str,
        choices=["annotation", "auto"],
        default="annotation",
        help="Use episode annotations when available, otherwise fall back to automatic captioning.",
    )
    parser.add_argument("--seed-base", type=int, default=777, help="Base seed; frame_idx is added for deterministic per-frame seeds.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for inference.")
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16", "float32"], default="bfloat16", help="Torch dtype for the pipeline.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen-Image-Layered", help="Model ID to load from Hugging Face.")
    parser.add_argument("--enable-model-cpu-offload", action="store_true", help="Use diffusers CPU offload instead of moving the full pipeline to GPU.")
    parser.add_argument("--enable-vae-tiling", action="store_true", help="Enable VAE tiling if supported by the pipeline.")
    parser.add_argument("--preview-max-side", type=int, default=448, help="Maximum side length for preview-grid cells.")
    parser.add_argument("--save-layer-frames", action="store_true", help="Also save each layer frame as PNG.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite ffmpeg outputs if they already exist.")
    parser.add_argument("--dry-run", action="store_true", help="List planned work without loading the model.")
    return parser.parse_args()


def import_qwen_runtime(dtype_name):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is not installed in .venv. Install PyTorch first, then install transformers>=4.51.3 and diffusers from git."
        ) from exc

    try:
        from diffusers import QwenImageLayeredPipeline
    except ImportError as exc:
        raise RuntimeError(
            "diffusers with QwenImageLayeredPipeline is not installed. Run: "
            "pip install git+https://github.com/huggingface/diffusers transformers>=4.51.3"
        ) from exc

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return torch, QwenImageLayeredPipeline, dtype_map[dtype_name]


def find_annotation_path(video_path, scan_root):
    scan_root = Path(scan_root).resolve()
    current = Path(video_path).resolve().parent
    while True:
        candidate = current / "annotation.hdf5"
        if candidate.exists():
            return candidate
        if current == scan_root or current.parent == current:
            return None
        current = current.parent


def load_image_names_for_annotation(annotation_path):
    with h5py.File(annotation_path, "r") as h5f:
        if "slam/frame_names" in h5f:
            frame_names_ds = h5f["slam/frame_names"]
            return [
                np.array(frame_names_ds[i]).tobytes().decode("utf-8", errors="replace").strip("\x00")
                for i in range(frame_names_ds.shape[0])
            ]

        num_frames = None
        for key in ("hand_mocap/left_joints_3d", "depth/depth", "full_body_mocap/keypoints"):
            if key in h5f:
                num_frames = int(h5f[key].shape[0])
                break
        if num_frames is None:
            return []
        return [f"frame_{idx:06d}.jpg" for idx in range(num_frames)]


def load_episode_caption_context(annotation_path):
    img_names = load_image_names_for_annotation(annotation_path)
    main_task, frame_info_map, _, _ = load_caption_data_from_annotation_hdf5(
        str(annotation_path),
        str(annotation_path.parent),
        img_names,
    )
    return EpisodeCaptionContext(
        annotation_path=Path(annotation_path),
        main_task=main_task or "",
        frame_info_map=frame_info_map,
    )


def build_prompt(frame_idx, args, caption_context):
    def _sentence(prefix, text):
        text = str(text).strip()
        if not text:
            return ""
        text = text.rstrip(". ")
        return f"{prefix}: {text}."

    if args.prompt is not None:
        return args.prompt.strip()

    if args.prompt_mode == "annotation" and caption_context and caption_context.frame_info_map:
        info = caption_context.frame_info_map.get(frame_idx, {})
        parts = []
        if caption_context.main_task:
            parts.append(_sentence("Main task", caption_context.main_task))
        theme = info.get("theme")
        if theme:
            parts.append(_sentence("Sub task", theme))
        action_label = info.get("action_label")
        if action_label:
            parts.append(_sentence("Action", action_label))
        action_desc = info.get("action_desc")
        if action_desc:
            parts.append(_sentence("Action detail", action_desc))
        objects = info.get("objects")
        if objects:
            objects_text = ", ".join(str(obj) for obj in objects if str(obj).strip())
            if objects_text:
                parts.append(_sentence("Objects", objects_text))
        prompt = " ".join(parts).strip()
        if prompt:
            return prompt

    return ""


def relative_video_output_dir(video_path, scan_root, output_root):
    video_path = Path(video_path).resolve()
    scan_root = Path(scan_root).resolve()
    try:
        relative_path = video_path.relative_to(scan_root)
    except ValueError:
        relative_path = Path(video_path.name)
    return Path(output_root).resolve() / relative_path.parent / video_path.stem


def select_videos(scan_root, include_globs, max_videos):
    scan_root = Path(scan_root).resolve()
    videos = discover_video_files(scan_root)
    if include_globs:
        filtered = []
        for video in videos:
            relative = video.resolve().relative_to(scan_root).as_posix()
            if any(Path(relative).match(pattern) for pattern in include_globs):
                filtered.append(video)
        videos = filtered
    if max_videos is not None:
        videos = videos[:max_videos]
    return videos


def compute_frame_schedule(total_frames, start_frame, end_frame, frame_stride, max_frames):
    start_frame = max(0, start_frame)
    stop_frame = total_frames if end_frame is None else min(end_frame, total_frames)
    if stop_frame < start_frame:
        stop_frame = start_frame
    indices = list(range(start_frame, stop_frame, frame_stride))
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def create_checkerboard(size, tile_size=24):
    width, height = size
    board = np.zeros((height, width, 4), dtype=np.uint8)
    light = np.array([238, 238, 238, 255], dtype=np.uint8)
    dark = np.array([190, 190, 190, 255], dtype=np.uint8)
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            color = light if ((top // tile_size) + (left // tile_size)) % 2 == 0 else dark
            board[top : top + tile_size, left : left + tile_size] = color
    return Image.fromarray(board, mode="RGBA")


def fit_size(size, max_side):
    width, height = size
    largest_side = max(width, height)
    if largest_side <= max_side:
        return width, height
    scale = max_side / float(largest_side)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def render_preview_grid(original_rgba, layer_images, max_side):
    cell_size = fit_size(layer_images[0].size, max_side)
    checkerboard = create_checkerboard(cell_size)

    preview_cells = [original_rgba.resize(cell_size, Image.Resampling.LANCZOS).convert("RGB")]
    for layer_image in layer_images:
        layer_resized = layer_image.resize(cell_size, Image.Resampling.LANCZOS).convert("RGBA")
        preview_cells.append(Image.alpha_composite(checkerboard.copy(), layer_resized).convert("RGB"))

    columns = min(3, len(preview_cells))
    rows = int(math.ceil(len(preview_cells) / columns))
    padding = 16
    label_height = 28
    grid_width = columns * cell_size[0] + (columns + 1) * padding
    grid_height = rows * (cell_size[1] + label_height) + (rows + 1) * padding
    grid = Image.new("RGB", (grid_width, grid_height), color=(20, 24, 28))
    draw = ImageDraw.Draw(grid)

    labels = ["original"] + [f"layer_{idx:02d}" for idx in range(len(layer_images))]
    for idx, (label, cell) in enumerate(zip(labels, preview_cells)):
        row = idx // columns
        col = idx % columns
        x = padding + col * (cell_size[0] + padding)
        y = padding + row * (cell_size[1] + label_height + padding)
        grid.paste(cell, (x, y + label_height))
        draw.text((x, y), label, fill=(240, 240, 240))
    return grid


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        for item in items:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_pipeline(args):
    torch, pipeline_cls, torch_dtype = import_qwen_runtime(args.dtype)
    pipeline = pipeline_cls.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    if args.enable_vae_tiling and hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()
    if args.enable_model_cpu_offload:
        if not args.device.startswith("cuda"):
            raise ValueError("--enable-model-cpu-offload requires a CUDA device.")
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(args.device)
    pipeline.set_progress_bar_config(disable=None)
    return torch, pipeline


def close_writers(writers, abort=False):
    first_error = None
    for writer in writers:
        try:
            writer.close(abort=abort)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            if first_error is None:
                first_error = exc
    if first_error is not None and not abort:
        raise first_error


def process_video(video_path, args, torch, pipeline, caption_context_cache):
    metadata = get_video_metadata(video_path)
    frame_indices = compute_frame_schedule(
        total_frames=metadata.num_frames,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    if not frame_indices:
        return {
            "video": str(video_path),
            "status": "skipped_no_frames",
            "processed_frames": 0,
        }

    video_output_dir = relative_video_output_dir(video_path, args.scan_root, args.output_root)
    manifest_path = video_output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite and not args.dry_run:
        return {
            "video": str(video_path),
            "status": "skipped_existing_manifest",
            "processed_frames": 0,
            "output_dir": str(video_output_dir),
        }

    annotation_path = find_annotation_path(video_path, args.scan_root)
    caption_context = None
    if annotation_path is not None:
        cache_key = str(annotation_path.resolve())
        if cache_key not in caption_context_cache:
            caption_context_cache[cache_key] = load_episode_caption_context(annotation_path)
        caption_context = caption_context_cache[cache_key]

    if args.dry_run:
        return {
            "video": str(video_path),
            "status": "dry_run",
            "output_dir": str(video_output_dir),
            "planned_frames": len(frame_indices),
            "annotation_path": str(annotation_path) if annotation_path else None,
            "prompt_preview": build_prompt(frame_indices[0], args, caption_context),
        }

    video_output_dir.mkdir(parents=True, exist_ok=True)
    prompt_records = []
    preview_writer = None
    layer_writers = []
    layer_frame_dirs = []
    inferred_layer_count = None
    processed_frames = 0
    error = None

    layer_video_paths = []
    preview_path = video_output_dir / "preview_grid.mp4"
    selected_frame_set = set(frame_indices)
    frame_iter = iter_video_frames(
        video_path,
        start_frame=frame_indices[0],
        end_frame=frame_indices[-1] + 1,
        frame_stride=1,
    )

    try:
        progress = None
        progress = tqdm(
            total=len(frame_indices),
            desc=Path(video_path).name,
            unit="frame",
            dynamic_ncols=True,
        )
        for frame_idx, frame_rgb in frame_iter:
            if frame_idx not in selected_frame_set:
                continue

            prompt = build_prompt(frame_idx, args, caption_context)
            prompt_records.append(
                {
                    "frame_idx": frame_idx,
                    "prompt": prompt,
                }
            )

            pil_frame = Image.fromarray(frame_rgb).convert("RGBA")
            generator = None
            if args.seed_base is not None:
                generator_device = args.device if args.device.startswith("cuda") else "cpu"
                generator = torch.Generator(device=generator_device).manual_seed(args.seed_base + frame_idx)

            with torch.inference_mode():
                output = pipeline(
                    image=pil_frame,
                    prompt=prompt,
                    generator=generator,
                    true_cfg_scale=args.true_cfg_scale,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.num_inference_steps,
                    num_images_per_prompt=1,
                    layers=args.layers,
                    resolution=args.resolution,
                    cfg_normalize=args.cfg_normalize,
                    use_en_prompt=args.use_en_prompt,
                )

            layer_images = [img.convert("RGBA") for img in output.images[0]]
            if not layer_images:
                raise RuntimeError(f"Pipeline returned no layers for frame {frame_idx} in {video_path}")

            if inferred_layer_count is None:
                inferred_layer_count = len(layer_images)
                layer_video_paths = [
                    video_output_dir / f"layer_{layer_idx:02d}_alpha.mov"
                    for layer_idx in range(inferred_layer_count)
                ]
                layer_writers = [
                    FFmpegRawVideoWriter(
                        output_path=path,
                        width=layer_images[0].width,
                        height=layer_images[0].height,
                        fps=metadata.fps,
                        input_pix_fmt="rgba",
                        codec_args=[
                            "-c:v",
                            "prores_ks",
                            "-profile:v",
                            "4",
                            "-pix_fmt",
                            "yuva444p10le",
                        ],
                        overwrite=args.overwrite,
                    )
                    for path in layer_video_paths
                ]
                if args.save_layer_frames:
                    layer_frame_dirs = [
                        video_output_dir / "frames" / f"layer_{layer_idx:02d}"
                        for layer_idx in range(inferred_layer_count)
                    ]
                    for directory in layer_frame_dirs:
                        directory.mkdir(parents=True, exist_ok=True)

                preview_frame = render_preview_grid(pil_frame, layer_images, args.preview_max_side)
                preview_writer = FFmpegRawVideoWriter(
                    output_path=preview_path,
                    width=preview_frame.width,
                    height=preview_frame.height,
                    fps=metadata.fps,
                    input_pix_fmt="rgb24",
                    codec_args=[
                        "-c:v",
                        "libx264",
                        "-crf",
                        "18",
                        "-preset",
                        "medium",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                    ],
                    overwrite=args.overwrite,
                )
                preview_writer.write(np.asarray(preview_frame, dtype=np.uint8))
            else:
                if len(layer_images) != inferred_layer_count:
                    raise RuntimeError(
                        f"Inconsistent layer count for {video_path}: expected {inferred_layer_count}, got {len(layer_images)}"
                    )
                preview_frame = render_preview_grid(pil_frame, layer_images, args.preview_max_side)
                preview_writer.write(np.asarray(preview_frame, dtype=np.uint8))

            for layer_idx, layer_image in enumerate(layer_images):
                layer_array = np.asarray(layer_image, dtype=np.uint8)
                layer_writers[layer_idx].write(layer_array)
                if args.save_layer_frames:
                    layer_image.save(layer_frame_dirs[layer_idx] / f"frame_{frame_idx:06d}.png")

            processed_frames += 1
            progress.update(1)
            if args.device.startswith("cuda") and processed_frames % 10 == 0:
                torch.cuda.empty_cache()
    except Exception as exc:
        error = exc
    finally:
        if progress is not None:
            progress.close()
        writers = list(layer_writers)
        if preview_writer is not None:
            writers.append(preview_writer)
        close_writers(writers, abort=error is not None)

    if error is not None:
        raise error

    manifest = {
        "source_video": str(Path(video_path).resolve()),
        "annotation_path": str(annotation_path.resolve()) if annotation_path else None,
        "output_dir": str(video_output_dir),
        "video_metadata": {
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
            "num_frames": metadata.num_frames,
            "duration_seconds": metadata.duration_seconds,
        },
        "frame_schedule": {
            "start_frame": frame_indices[0],
            "end_frame_inclusive": frame_indices[-1],
            "frame_stride": args.frame_stride,
            "planned_frames": len(frame_indices),
            "processed_frames": processed_frames,
        },
        "pipeline": {
            "model_id": args.model_id,
            "device": args.device,
            "dtype": args.dtype,
            "layers_requested": args.layers,
            "layers_emitted": inferred_layer_count,
            "resolution": args.resolution,
            "num_inference_steps": args.num_inference_steps,
            "true_cfg_scale": args.true_cfg_scale,
            "negative_prompt": args.negative_prompt,
            "cfg_normalize": args.cfg_normalize,
            "use_en_prompt": args.use_en_prompt,
            "seed_base": args.seed_base,
        },
        "outputs": {
            "preview_grid_mp4": str(preview_path),
            "layer_alpha_movs": [str(path) for path in layer_video_paths],
            "prompt_records_jsonl": str(video_output_dir / "prompts.jsonl"),
            "saved_layer_frames": args.save_layer_frames,
        },
        "prompt_mode": args.prompt_mode if args.prompt is None else "fixed",
        "fixed_prompt": args.prompt,
        "main_task": caption_context.main_task if caption_context else "",
    }
    write_json(manifest_path, manifest)
    write_jsonl(video_output_dir / "prompts.jsonl", prompt_records)
    return {
        "video": str(video_path),
        "status": "completed",
        "processed_frames": processed_frames,
        "output_dir": str(video_output_dir),
    }


def main():
    args = parse_args()
    videos = select_videos(args.scan_root, args.include_glob, args.max_videos)
    if not videos:
        print(f"No videos found under {Path(args.scan_root).resolve()}")
        return 1

    print(f"Discovered {len(videos)} video(s) under {Path(args.scan_root).resolve()}")
    for video in videos:
        metadata = get_video_metadata(video)
        planned_frames = compute_frame_schedule(
            total_frames=metadata.num_frames,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
        )
        print(
            f"- {video}: {metadata.width}x{metadata.height}, fps={metadata.fps:.3f}, "
            f"frames={metadata.num_frames}, planned={len(planned_frames)}"
        )

    if args.dry_run:
        caption_context_cache = {}
        dry_run_results = [process_video(video, args, None, None, caption_context_cache) for video in videos]
        print(json.dumps(dry_run_results, indent=2, ensure_ascii=False))
        return 0

    torch, pipeline = load_pipeline(args)
    caption_context_cache = {}
    results = []
    for video in videos:
        result = process_video(video, args, torch, pipeline, caption_context_cache)
        results.append(result)

    completed = [item for item in results if item["status"] == "completed"]
    skipped = [item for item in results if item["status"] != "completed"]
    print(f"Completed {len(completed)} video(s); skipped {len(skipped)}.")
    for item in results:
        print(json.dumps(item, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
