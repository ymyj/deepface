"""Flask web application – Demo UI + REST API in one server."""

import os, re, uuid, base64, logging, json
import cv2
import numpy as np
from flask import (Flask, request, jsonify, render_template,
                   send_file, send_from_directory, Response, stream_with_context)

from typing import Optional
from .recognizer import FaceRecognizer
from .video_processor import VideoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(recognizer: Optional[FaceRecognizer] = None,
               upload_dir: Optional[str] = None,
               output_dir: Optional[str] = None) -> Flask:
    """Build and return a Flask application instance.

    Args:
        recognizer: Shared FaceRecognizer (created fresh if None).
        upload_dir: Directory for uploaded files.
        output_dir: Directory for processed output videos.
    """
    app = Flask(__name__)

    # Directories
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app.config["UPLOAD_DIR"] = upload_dir or os.path.join(base, "uploads")
    app.config["OUTPUT_DIR"] = output_dir or os.path.join(base, "outputs")
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

    os.makedirs(app.config["UPLOAD_DIR"], exist_ok=True)
    os.makedirs(app.config["OUTPUT_DIR"], exist_ok=True)

    # Shared recogniser
    DETECTOR = "ssd"  # ssd (DNN, accurate), opencv (Haar, often fails), mtcnn, retinaface
    recognizer_ = recognizer or FaceRecognizer(
        model_name="Facenet512",
        detector_backend=DETECTOR,
    )
    # Video processor — same detector for consistency
    video_processor_ = VideoProcessor(
        recognizer_,
        process_every_n_frames=10,
        detector_backend=DETECTOR,
    )

    # Register built-in template directory
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    app.template_folder = template_dir

    # ------------------------------------------------------------------
    # Web demo page
    # ------------------------------------------------------------------
    @app.route("/")
    def index():
        return render_template("index.html")

    # ------------------------------------------------------------------
    # API: Register a face
    # ------------------------------------------------------------------
    @app.route("/api/register", methods=["POST"])
    def api_register():
        name = request.form.get("name", "").strip()
        if not name:
            return jsonify({"status": "error", "message": "Missing name"}), 400

        file = request.files.get("image")
        if not file:
            return jsonify({"status": "error", "message": "Missing image"}), 400

        # Use UUID as filename to avoid Unicode path issues with OpenCV
        ext = os.path.splitext(file.filename or ".jpg")[1] or ".jpg"
        safe_name = uuid.uuid4().hex
        dest = os.path.join(app.config["UPLOAD_DIR"], f"ref_{safe_name}{ext}")
        file.save(dest)

        result = recognizer_.register(name, dest)
        if result["status"] == "error":
            return jsonify(result), 400
        return jsonify(result), 200

    # ------------------------------------------------------------------
    # API: List registered faces
    # ------------------------------------------------------------------
    @app.route("/api/faces", methods=["GET"])
    def api_faces():
        faces = recognizer_.list_faces()
        return jsonify({"faces": faces}), 200

    # ------------------------------------------------------------------
    # API: Get face image
    # ------------------------------------------------------------------
    @app.route("/api/face-image/<name>")
    def api_face_image(name):
        faces = recognizer_.list_faces()
        for f in faces:
            if f["name"] == name:
                return send_file(f["image_path"])
        return jsonify({"error": "not found"}), 404

    # ------------------------------------------------------------------
    # API: Delete a registered face
    # ------------------------------------------------------------------
    @app.route("/api/face/<name>", methods=["DELETE"])
    def api_delete_face(name):
        ok = recognizer_.remove_face(name)
        if ok:
            return jsonify({"status": "ok"}), 200
        return jsonify({"status": "error", "message": "not found"}), 404

    # ------------------------------------------------------------------
    # API: Verify two uploaded images
    # ------------------------------------------------------------------
    @app.route("/api/verify", methods=["POST"])
    def api_verify():
        f1 = request.files.get("image1")
        f2 = request.files.get("image2")
        if not f1 or not f2:
            return jsonify({"verified": False, "error": "Need two images"}), 400

        p1 = _save_upload(f1, app.config["UPLOAD_DIR"])
        p2 = _save_upload(f2, app.config["UPLOAD_DIR"])
        result = recognizer_.verify_pair(p1, p2)
        return jsonify(result), 200

    # ------------------------------------------------------------------
    # API: Recognize faces in an uploaded image
    # ------------------------------------------------------------------
    @app.route("/api/recognize", methods=["POST"])
    def api_recognize():
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image"}), 400

        path = _save_upload(file, app.config["UPLOAD_DIR"])
        matches = recognizer_.recognize(path)
        return jsonify({"matches": matches}), 200

    # ------------------------------------------------------------------
    # API: Process a video for face recognition
    # ------------------------------------------------------------------
    @app.route("/api/recognize-video", methods=["POST"])
    def api_recognize_video():
        file = request.files.get("video")
        if not file:
            return jsonify({"status": "error", "message": "No video file"}), 400

        # Save uploaded video
        ext = os.path.splitext(file.filename or ".mp4")[1] or ".mp4"
        uid = uuid.uuid4().hex
        video_path = os.path.join(app.config["UPLOAD_DIR"], f"vid_{uid}{ext}")
        file.save(video_path)

        # Process
        out_name = f"out_{uid}.mp4"
        out_path = os.path.join(app.config["OUTPUT_DIR"], out_name)

        try:
            result = video_processor_.process_video(
                video_path=video_path,
                output_path=out_path,
                show_progress=True,
            )
        except Exception as e:
            logger.exception("Video processing failed")
            return jsonify({"status": "error", "message": str(e)}), 500

        result["status"] = "ok"

        # Provide URL for the output video
        rel = f"/outputs/{out_name}"
        result["output_video_url"] = rel

        return jsonify(result), 200

    # ------------------------------------------------------------------
    # API: Process a video with SSE streaming (real-time preview)
    # ------------------------------------------------------------------
    @app.route("/api/recognize-video-stream", methods=["POST"])
    def api_recognize_video_stream():
        file = request.files.get("video")
        if not file:
            return jsonify({"status": "error", "message": "No video file"}), 400

        # Save uploaded video
        ext = os.path.splitext(file.filename or ".mp4")[1] or ".mp4"
        uid = uuid.uuid4().hex
        video_path = os.path.join(app.config["UPLOAD_DIR"], f"vid_{uid}{ext}")
        file.save(video_path)

        def generate():
            """Generate SSE events for each processed frame."""
            try:
                for event in video_processor_.process_video_stream(video_path, preview_width=480):
                    sse_data = f"data: {json.dumps(event)}\n\n"
                    yield sse_data.encode('utf-8')
            except Exception as e:
                logger.exception("Stream processing failed")
                error_event = json.dumps({"error": str(e), "done": True})
                yield f"data: {error_event}\n\n".encode('utf-8')

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            }
        )

    # ------------------------------------------------------------------
    # API: Recognize faces from a base64 camera frame (real-time)
    # ------------------------------------------------------------------
    @app.route("/api/recognize-frame", methods=["POST"])
    def api_recognize_frame():
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"matches": []}), 400

        try:
            # Strip data:image/jpeg;base64, prefix
            b64 = re.sub(r"^data:image/\w+;base64,", "", data["image"])
            img_bytes = base64.b64decode(b64)
            img_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({"matches": []}), 400

            tmp = os.path.join(app.config["UPLOAD_DIR"],
                               f"_cam_{uuid.uuid4().hex}.jpg")
            cv2.imwrite(tmp, img)
            try:
                matches = recognizer_.recognize(tmp)
            finally:
                os.remove(tmp)

            return jsonify({"matches": matches}), 200
        except Exception as e:
            logger.warning("Frame recognition failed: %s", e)
            return jsonify({"matches": []}), 200

    # ------------------------------------------------------------------
    # Serve output videos
    # ------------------------------------------------------------------
    @app.route("/outputs/<filename>")
    def serve_output(filename):
        return send_from_directory(app.config["OUTPUT_DIR"], filename)

    return app


def _save_upload(file, directory: str) -> str:
    uid = uuid.uuid4().hex
    ext = os.path.splitext(file.filename or ".jpg")[1] or ".jpg"
    path = os.path.join(directory, f"{uid}{ext}")
    file.save(path)
    return path
