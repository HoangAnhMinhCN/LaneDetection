"""Local Gradio UI for the existing OpenCV lane detector."""

import csv
import json
import math
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr
import imageio_ffmpeg

from lane_detection import LaneDetector

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
FRAME_FIELDS = [
    "frame", "timestamp_s", "processing_ms", "left_observed",
    "right_observed", "left_tracked", "right_tracked",
]


def analyze_video(video, proc_width=960, smooth=0.9, length=0.75, progress=gr.Progress()):
    if not video or not Path(video).is_file():
        raise gr.Error("Hãy tải lên một clip trước khi chạy nhận diện.")

    started = time.perf_counter()
    cap = cv2.VideoCapture(str(video))
    writer = None
    try:
        if not cap.isOpened():
            raise gr.Error("Không mở được video. Hãy thử clip MP4 với codec H.264.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps_fallback = not math.isfinite(fps) or fps <= 0
        if fps_fallback:
            fps = 30.0
        reported_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total = max(0, int(reported_total)) if math.isfinite(reported_total) else 0
        ok, frame = cap.read()
        if not ok:
            raise gr.Error("Video không có frame đọc được.")
        height, width = frame.shape[:2]
        RUNS.mkdir(exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="clip_", dir=RUNS))
        raw = work / "overlay.mp4"
        output = work / "detected.mp4"
        csv_path = work / "frames.csv"
        json_path = work / "summary.json"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise gr.Error("Không tạo được video đầu ra trên máy này.")

        detector = LaneDetector(proc_width=int(proc_width), smooth=float(smooth), length=float(length))
        count = left_count = right_count = both_count = tracked_count = 0
        detection_seconds = 0.0
        progress(0, desc="Đang nhận diện làn đường…")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            rows = csv.DictWriter(handle, fieldnames=FRAME_FIELDS)
            rows.writeheader()
            while ok:
                tick = time.perf_counter()
                result = detector.process(frame)
                elapsed = time.perf_counter() - tick
                detection_seconds += elapsed
                writer.write(result)
                left = detector.left_observed
                right = detector.right_observed
                tracked = detector.left is not None and detector.right is not None
                rows.writerow({
                    "frame": count, "timestamp_s": round(count / fps, 6),
                    "processing_ms": round(elapsed * 1000, 3),
                    "left_observed": int(left), "right_observed": int(right),
                    "left_tracked": int(detector.left is not None),
                    "right_tracked": int(detector.right is not None),
                })
                count += 1
                left_count += int(left)
                right_count += int(right)
                both_count += int(left and right)
                tracked_count += int(tracked)
                if count == 1 or count % 15 == 0:
                    progress(min(0.85, 0.85 * count / total) if total else 0,
                             desc=f"Đã xử lý {count:,}/{total or '?'} frame")
                ok, frame = cap.read()

        writer.release()
        writer = None
        cap.release()
        processing_seconds = time.perf_counter() - started
        progress(0.88, desc="Đang xuất MP4 H.264 và giữ âm thanh…")
        # imageio-ffmpeg supplies an executable, so no system FFmpeg install is needed.
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error",
            "-i", str(raw), "-i", str(video), "-map", "0:v:0", "-map", "1:a?",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-movflags", "+faststart", str(output),
        ]
        encoded = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if encoded.returncode or not output.is_file() or output.stat().st_size == 0:
            raise gr.Error(f"Không mã hóa được MP4: {encoded.stderr[-1200:]}")
        raw.unlink()
        wall_seconds = time.perf_counter() - started
        summary = {
            "input_name": Path(video).name,
            "width": width, "height": height, "source_fps": round(fps, 3),
            "source_fps_fallback": fps_fallback, "frames_processed": count,
            "reported_frame_count": total, "duration_s": round(count / fps, 3),
            "processing_seconds": round(processing_seconds, 3),
            "total_seconds_including_encoding": round(wall_seconds, 3),
            "pipeline_fps": round(count / processing_seconds, 2),
            "detector_fps": round(count / detection_seconds, 2) if detection_seconds else 0,
            "mean_detector_ms": round(detection_seconds * 1000 / count, 2),
            "left_observed_frames": left_count, "right_observed_frames": right_count,
            "both_observed_frames": both_count, "both_tracked_frames": tracked_count,
            "left_observed_percent": round(100 * left_count / count, 2),
            "right_observed_percent": round(100 * right_count / count, 2),
            "both_observed_percent": round(100 * both_count / count, 2),
            "both_tracked_percent": round(100 * tracked_count / count, 2),
            "settings": {"proc_width": int(proc_width), "smooth": float(smooth), "length": float(length)},
            "note": "Observed = new segments accepted in this frame; tracked includes previous estimates. These are coverage statistics, not accuracy or confidence scores.",
        }
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        cards = "<div class='metrics'>" + "".join(
            f"<div class='metric'><span>{label}</span><strong>{value}</strong></div>"
            for label, value in [
                ("Tìm mới cả hai vạch", f"{summary['both_observed_percent']:.1f}%"),
                ("Hiển thị đủ hai làn", f"{summary['both_tracked_percent']:.1f}%"),
                ("Tốc độ pipeline", f"{summary['pipeline_fps']:.1f} FPS"),
                ("Tổng thời gian", f"{wall_seconds:.1f} s"),
            ]
        ) + "</div>"
        table = [
            ["Độ phân giải đầu vào", f"{width} × {height}"],
            ["FPS video", f"{fps:.3f}" + (" (dùng mặc định)" if fps_fallback else "")],
            ["Thời lượng đã xử lý", f"{count / fps:.2f} s"],
            ["Số frame đã xử lý", str(count)],
            ["Thời gian detector / frame", f"{summary['mean_detector_ms']:.2f} ms"],
            ["Frame tìm mới vạch trái", f"{left_count:,} ({summary['left_observed_percent']:.1f}%)"],
            ["Frame tìm mới vạch phải", f"{right_count:,} ({summary['right_observed_percent']:.1f}%)"],
            ["Frame tìm mới cả hai vạch", f"{both_count:,} ({summary['both_observed_percent']:.1f}%)"],
            ["Frame hiển thị đủ hai làn", f"{tracked_count:,} ({summary['both_tracked_percent']:.1f}%)"],
        ]
        status = "Hoàn tất. Có thể xem và tải clip kết quả bên dưới."
        if total > count:
            status += f" Lưu ý: chỉ đọc được {count}/{total} frame theo metadata của clip."
        progress(1, desc="Hoàn tất")
        return str(output), cards, table, [str(csv_path), str(json_path)], status
    finally:
        cap.release()
        if writer is not None:
            writer.release()


CSS = """
.gradio-container {max-width: 1240px !important;}
#hero {padding: 24px 0 12px;}
#hero h1 {font-size: 34px; margin-bottom: 8px;}
#hero p {color: #64748b; font-size: 16px;}
.metrics {display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 8px 0;}
.metric {padding: 20px; border: 1px solid #cbd5e1; border-radius: 14px;}
.metric span {display: block; font-size: 13px; color: #64748b;}
.metric strong {display: block; font-size: 27px; margin-top: 6px; color: #0d9488;}
@media (max-width: 700px) {.metrics {grid-template-columns: repeat(2, 1fr);}}
"""


def build_app():
    with gr.Blocks(title="Lane Detection Studio") as demo:
        gr.HTML("<div id='hero'><h1>Lane Detection Studio</h1>"
                "<p>Tải clip đường đi. Nhận video đánh dấu làn đường và thống kê từng frame.</p></div>")
        with gr.Row(equal_height=True):
            with gr.Column():
                source = gr.Video(label="01 · Clip đầu vào", sources=["upload"], height=320)
                with gr.Accordion("Tùy chỉnh nhận diện", open=False):
                    proc_width = gr.Slider(320, 1280, value=960, step=160, label="Chiều rộng xử lý (px)",
                                           info="Giảm để chạy nhanh hơn. Video đầu ra giữ độ phân giải gốc.")
                    smooth = gr.Slider(0, 0.99, value=0.9, step=0.01, label="Độ làm mượt theo thời gian")
                    length = gr.Slider(0.1, 1, value=0.75, step=0.05, label="Độ dài vạch hiển thị")
                run = gr.Button("Chạy nhận diện", variant="primary", size="lg")
            with gr.Column():
                result = gr.Video(label="02 · Video đã nhận diện", height=320, interactive=False)
                status = gr.Textbox(label="Trạng thái", value="Tải clip lên để bắt đầu.", interactive=False)
        samples = sorted((ROOT / "Input").glob("*.mp4"))
        if samples:
            gr.Examples(examples=[[str(path)] for path in samples], inputs=[source], label="Hoặc chọn clip có sẵn")
        gr.Markdown("### Thống kê xử lý")
        metrics = gr.HTML()
        with gr.Row():
            details = gr.Dataframe(headers=["Chỉ số", "Giá trị"], datatype=["str", "str"],
                                   interactive=False, label="Chi tiết")
            downloads = gr.File(label="Tải thống kê · CSV từng frame / JSON tổng hợp", file_count="multiple", interactive=False)
        gr.Markdown(
            "**Cách đọc số liệu:** Tìm mới = frame có đoạn vạch được chấp nhận sau lọc nhiễu. "
            "Hiển thị = gồm cả ước lượng giữ từ frame trước để làm mượt. "
            "Các tỷ lệ này đo độ phủ, không phải độ chính xác hoặc confidence. "
            "FPS pipeline gồm đọc, nhận diện và ghi frame; tổng thời gian gồm cả xuất H.264."
        )
        run.click(analyze_video, inputs=[source, proc_width, smooth, length],
                  outputs=[result, metrics, details, downloads, status], concurrency_limit=1,
                  api_name="detect")
    return demo


if __name__ == "__main__":
    build_app().queue(max_size=8).launch(
        server_name="127.0.0.1", server_port=7860,
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"), css=CSS,
        allowed_paths=[str(ROOT / "Input"), str(RUNS)],
    )
