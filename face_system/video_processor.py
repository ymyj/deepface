"""Video processing module: scan videos for registered faces."""

import os
import cv2
import numpy as np
import tempfile
import base64
import json
from deepface import DeepFace
from .recognizer import FaceRecognizer, cosine_distance
import logging

logger = logging.getLogger(__name__)

_COLORS = {
    "match": (0, 255, 0),
    "unknown": (0, 0, 255),
    "text_bg": (0, 0, 0),
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class VideoProcessor:
    """Process video files to detect and recognise faces.

    Uses DeepFace.represent() directly on each frame for consistent
    high-accuracy face detection + embedding extraction (not Haar cascade).
    """

    def __init__(self, recognizer: FaceRecognizer,
                 process_every_n_frames: int = 10,
                 detector_backend: str = "opencv"):
        """
        Args:
            recognizer: Initialised FaceRecognizer with registered faces.
            process_every_n_frames: Process every N frames.
            detector_backend: DeepFace detector backend.
                              Use 'opencv' (fast), 'retinaface' (accurate),
                              'mtcnn' (balanced), or 'ssd' (fast).
        """
        self.recognizer = recognizer
        self.every_n = process_every_n_frames
        self.detector_backend = detector_backend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_video(self, video_path: str, output_path: str = None,
                      show_progress: bool = True) -> dict:
        """Scan a video for registered faces using DeepFace directly.

        Each sampled frame is analysed by DeepFace.represent() which
        handles both face detection and embedding extraction in one
        coherent pipeline — no separate Haar cascade step.

        Returns:
            dict with total_frames, fps, duration, matches, match_summary,
            output_video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        out_writer = None
        if output_path:
            # Try H.264 (avc1) for browser compatibility, fall back to mp4v
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out_writer = cv2.VideoWriter(output_path, fourcc, fps,
                                         (width, height))
            if not out_writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_writer = cv2.VideoWriter(output_path, fourcc, fps,
                                             (width, height))

        all_matches: list[dict] = []
        frame_idx = 0
        match_counts: dict[str, int] = {}

        tmp_dir = tempfile.mkdtemp(prefix="deepface_")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames for performance
                if frame_idx % self.every_n != 0:
                    if out_writer:
                        out_writer.write(frame)
                    frame_idx += 1
                    continue

                timestamp = frame_idx / fps if fps > 0 else 0

                # --- Use DeepFace directly on this frame ---
                frame_path = os.path.join(tmp_dir, f"f{frame_idx}.jpg")
                cv2.imwrite(frame_path, frame)

                try:
                    results = DeepFace.represent(
                        img_path=frame_path,
                        model_name=self.recognizer.model_name,
                        detector_backend=self.detector_backend,
                        enforce_detection=False,
                    )
                except Exception as e:
                    logger.warning("Frame %d represent failed: %s", frame_idx, e)
                    results = []

                for det in results:
                    emb = np.array(det["embedding"])
                    region = det.get("region", {})
                    x = region.get("x", 0)
                    y = region.get("y", 0)
                    w = region.get("w", 0)
                    h = region.get("h", 0)

                    # Match against all registered faces
                    best_name = "unknown"
                    best_dist = float("inf")
                    for name, (ref_emb, _) in self.recognizer.faces.items():
                        dist = cosine_distance(ref_emb, emb)
                        if dist < best_dist:
                            best_dist = dist
                            best_name = name if dist < self.recognizer.threshold else "unknown"

                    similarity = max(0.0, 1.0 - best_dist)
                    is_match = best_name != "unknown"

                    # Draw bounding box
                    color = _COLORS["match"] if is_match else _COLORS["unknown"]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    label = f"{best_name} ({similarity:.0%})" if is_match else "unknown"
                    (tw, th), _ = cv2.getTextSize(label, _FONT, 0.6, 1)
                    cv2.rectangle(frame, (x, y - th - 6),
                                  (x + tw + 4, y), _COLORS["text_bg"], -1)
                    cv2.putText(frame, label, (x + 2, y - 4),
                                _FONT, 0.6, color, 1)

                    if is_match:
                        all_matches.append({
                            "frame": frame_idx,
                            "timestamp": round(timestamp, 2),
                            "name": best_name,
                            "distance": round(best_dist, 4),
                            "similarity": round(similarity, 4),
                        })
                        match_counts[best_name] = match_counts.get(best_name, 0) + 1

                if show_progress and frame_idx % (self.every_n * 20) == 0:
                    pct = frame_idx / total_frames * 100
                    print(f"\r  Processing: {frame_idx}/{total_frames} "
                          f"({pct:.0f}%)", flush=True, end="")

                if out_writer:
                    out_writer.write(frame)
                frame_idx += 1

        finally:
            cap.release()
            if out_writer:
                out_writer.release()
            # Clean temp files
            for f_name in os.listdir(tmp_dir):
                try:
                    os.remove(os.path.join(tmp_dir, f_name))
                except OSError:
                    pass
            os.rmdir(tmp_dir)

        print()

        return {
            "total_frames": total_frames,
            "fps": fps,
            "duration": round(duration, 2),
            "matches": all_matches,
            "match_summary": match_counts,
            "output_video": output_path,
        }

    # ------------------------------------------------------------------
    # Streaming API (SSE)
    # ------------------------------------------------------------------
    def process_video_stream(self, video_path: str, preview_width: int = 480):
        """Process video and yield SSE events for real-time preview.

        Yields tuples of (frame_index, frame_data, detections, summary).
        frame_data is a base64-encoded JPEG of the annotated frame resized to preview_width.
        detections is a list of face detection results with thumbnail data.

        Args:
            video_path: Path to the input video.
            preview_width: Width of the preview frame (height auto-scaled).

        Yields:
            dict with frame_idx, total_frames, progress, fps, duration,
            frame_base64, detections, all_matches, match_summary.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        # Calculate preview dimensions
        scale = preview_width / orig_width if orig_width > 0 else 1
        preview_height = int(orig_height * scale)

        all_matches: list[dict] = []
        frame_idx = 0
        match_counts: dict[str, int] = {}
        processed_count = 0

        tmp_dir = tempfile.mkdtemp(prefix="deepface_stream_")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_idx / fps if fps > 0 else 0

                # Prepare detection data for this frame
                frame_detections = []

                if frame_idx % self.every_n == 0:
                    # --- Use DeepFace directly on this frame ---
                    frame_path = os.path.join(tmp_dir, f"f{frame_idx}.jpg")
                    cv2.imwrite(frame_path, frame)

                    try:
                        results = DeepFace.represent(
                            img_path=frame_path,
                            model_name=self.recognizer.model_name,
                            detector_backend=self.detector_backend,
                            enforce_detection=False,
                        )
                    except Exception as e:
                        logger.warning("Frame %d represent failed: %s", frame_idx, e)
                        results = []

                    for det in results:
                        emb = np.array(det["embedding"])
                        region = det.get("region", {})
                        x = region.get("x", 0)
                        y = region.get("y", 0)
                        w = region.get("w", 0)
                        h = region.get("h", 0)

                        # Match against all registered faces
                        best_name = "unknown"
                        best_dist = float("inf")
                        for name, (ref_emb, _) in self.recognizer.faces.items():
                            dist = cosine_distance(ref_emb, emb)
                            if dist < best_dist:
                                best_dist = dist
                                best_name = name if dist < self.recognizer.threshold else "unknown"

                        similarity = max(0.0, 1.0 - best_dist)
                        is_match = best_name != "unknown"

                        # Crop face region for thumbnail (64x64)
                        thumb_base64 = None
                        try:
                            x1, y1 = max(0, x), max(0, y)
                            x2, y2 = min(orig_width, x + w), min(orig_height, y + h)
                            if x2 > x1 and y2 > y1:
                                face_crop = frame[y1:y2, x1:x2]
                                face_crop = cv2.resize(face_crop, (64, 64))
                                _, buf = cv2.imencode('.jpg', face_crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
                                thumb_base64 = base64.b64encode(buf).decode('utf-8')
                        except Exception:
                            pass

                        # Draw bounding box on frame
                        color = _COLORS["match"] if is_match else _COLORS["unknown"]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        label = f"{best_name} ({similarity:.0%})" if is_match else "unknown"
                        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.6, 1)
                        cv2.rectangle(frame, (x, y - th - 6),
                                      (x + tw + 4, y), _COLORS["text_bg"], -1)
                        cv2.putText(frame, label, (x + 2, y - 4),
                                    _FONT, 0.6, color, 1)

                        det_info = {
                            "name": best_name,
                            "is_match": is_match,
                            "similarity": round(similarity, 4),
                            "distance": round(best_dist, 4),
                            "region": {"x": x, "y": y, "w": w, "h": h},
                            "thumbnail": thumb_base64,
                            "timestamp": round(timestamp, 2),
                            "frame": frame_idx,
                        }
                        frame_detections.append(det_info)

                        if is_match:
                            all_matches.append({
                                "frame": frame_idx,
                                "timestamp": round(timestamp, 2),
                                "name": best_name,
                                "distance": round(best_dist, 4),
                                "similarity": round(similarity, 4),
                                "thumbnail": thumb_base64,
                            })
                            match_counts[best_name] = match_counts.get(best_name, 0) + 1

                # Resize frame to preview size
                preview_frame = cv2.resize(frame, (preview_width, preview_height))
                _, buf = cv2.imencode('.jpg', preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                preview_base64 = base64.b64encode(buf).decode('utf-8')

                progress = (frame_idx + 1) / total_frames * 100 if total_frames > 0 else 0

                yield {
                    "frame_idx": frame_idx,
                    "total_frames": total_frames,
                    "progress": round(progress, 1),
                    "fps": fps,
                    "duration": duration,
                    "timestamp": round(timestamp, 2),
                    "frame_base64": preview_base64,
                    "detections": frame_detections,
                    "all_matches": all_matches,
                    "match_summary": match_counts,
                    "done": False,
                }

                processed_count += 1
                frame_idx += 1

        finally:
            cap.release()
            # Clean temp files
            for f_name in os.listdir(tmp_dir):
                try:
                    os.remove(os.path.join(tmp_dir, f_name))
                except OSError:
                    pass
            os.rmdir(tmp_dir)

        # Final result
        yield {
            "frame_idx": frame_idx,
            "total_frames": total_frames,
            "progress": 100.0,
            "fps": fps,
            "duration": duration,
            "all_matches": all_matches,
            "match_summary": match_counts,
            "done": True,
        }
