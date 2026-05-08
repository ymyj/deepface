"""Core face recognition module using DeepFace."""

import os
import pickle
import numpy as np
from deepface import DeepFace
import logging

logger = logging.getLogger(__name__)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two vectors (DeepFace default metric).

    Range: 0 = identical, 1 = orthogonal, 2 = opposite.
    """
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("inf")
    return float(1.0 - dot / (na * nb))


class FaceRecognizer:
    """Face recognition engine: register reference faces and match against them."""

    # Default thresholds per model (cosine distance)
    MODEL_THRESHOLDS = {
        "Facenet512": 0.30,
        "VGG-Face": 0.40,
        "ArcFace": 0.40,
        "Facenet": 0.40,
        "DeepFace": 0.25,
        "DeepID": 0.015,
        "Dlib": 0.07,
        "SFace": 0.30,
        "GhostFaceNet": 0.40,
        "OpenFace": 0.10,
    }

    def __init__(self, model_name="Facenet512", detector_backend="opencv",
                 threshold=None, db_path=None):
        """
        Args:
            model_name: DeepFace model (Facenet512, VGG-Face, ArcFace, …)
            detector_backend: Face detector (opencv, mtcnn, retinaface, ssd)
            threshold: Cosine distance threshold (None = use model default)
            db_path: Path to persist enrolled face embeddings.
        """
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.threshold = threshold or self.MODEL_THRESHOLDS.get(model_name, 0.30)
        self.db_path = db_path or os.path.join(os.path.dirname(__file__), "face_db")
        os.makedirs(self.db_path, exist_ok=True)

        # Maps name -> (embedding, image_path)
        self.faces: dict[str, tuple[np.ndarray, str]] = {}
        self._load_faces()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self, name: str, img_path: str) -> dict:
        """Register a person by name with a reference image."""
        try:
            results = DeepFace.represent(
                img_path=img_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True,
            )
            if not results:
                raise ValueError("No face detected in the image")

            embedding = np.array(results[0]["embedding"])
            face_area = results[0].get("area", 0)
            confidence = results[0].get("confidence", 0)

            self.faces[name] = (embedding, img_path)
            self._save_face(name, embedding, img_path)

            return {
                "status": "ok",
                "name": name,
                "face_confidence": float(confidence),
                "face_area": int(face_area),
                "threshold": self.threshold,
            }
        except Exception as e:
            logger.error("Failed to register %s: %s", name, e)
            return {"status": "error", "name": name, "message": str(e)}

    def list_faces(self) -> list[dict]:
        """Return list of registered face names and metadata."""
        return [
            {"name": name, "image_path": path}
            for name, (_, path) in self.faces.items()
        ]

    def remove_face(self, name: str) -> bool:
        """Remove a registered face."""
        if name in self.faces:
            del self.faces[name]
            db_file = os.path.join(self.db_path, f"{name}.pkl")
            if os.path.exists(db_file):
                os.remove(db_file)
            return True
        return False

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------
    def recognize(self, img_path: str) -> list[dict]:
        """Find all faces in an image and match against registered faces.

        Returns list of dicts: [{name, distance, similarity, region, confidence}]
        """
        if not self.faces:
            return [{"name": "no_faces_registered"}]

        try:
            results = DeepFace.represent(
                img_path=img_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
            )
        except Exception as e:
            logger.warning("Face detection failed: %s", e)
            return []

        matches = []
        for det in results:
            probe_emb = np.array(det["embedding"])
            best_name, best_dist = self._find_best_match(probe_emb)
            region = det.get("region", {})

            matches.append({
                "name": best_name,
                "distance": round(best_dist, 4),
                "similarity": round(max(0.0, 1.0 - best_dist), 4),
                "is_match": bool(best_dist < self.threshold),
                "region": region,
                "confidence": float(det.get("confidence", 0)),
            })
        return matches

    def verify_pair(self, img1_path: str, img2_path: str) -> dict:
        """Directly verify two faces using DeepFace.verify (uses correct metric)."""
        try:
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
            )
            dist = float(result.get("distance", 1.0))
            return {
                "verified": bool(result.get("verified", False)),
                "distance": round(dist, 4),
                "threshold": float(result.get("threshold", self.threshold)),
                "similarity": round(max(0.0, 1.0 - dist), 4),
                "model": self.model_name,
                "similarity_metric": "cosine",
            }
        except Exception as e:
            logger.error("Verify failed: %s", e)
            return {"verified": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _find_best_match(self, probe_emb: np.ndarray) -> tuple[str, float]:
        best_name = "unknown"
        best_dist = float("inf")
        for name, (ref_emb, _) in self.faces.items():
            dist = cosine_distance(ref_emb, probe_emb)
            if dist < best_dist:
                best_dist = dist
                best_name = name if dist < self.threshold else "unknown"
        return best_name, best_dist

    def _save_face(self, name: str, embedding: np.ndarray, img_path: str):
        path = os.path.join(self.db_path, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump({"embedding": embedding, "image_path": img_path}, f)

    def _load_faces(self):
        for fname in os.listdir(self.db_path):
            if fname.endswith(".pkl"):
                name = fname[:-4]
                path = os.path.join(self.db_path, fname)
                try:
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                    self.faces[name] = (data["embedding"], data["image_path"])
                except Exception as e:
                    logger.warning("Failed to load %s: %s", fname, e)
