from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp


@dataclass
class VerificationResult:
    matched: bool
    distance: float
    liveness: str = "LIVE"
    confidence: float = 0.0
    identity_id: str | None = None
    identity_name: str | None = None


@dataclass
class FrameAssessment:
    ready: bool
    message: str


@dataclass
class FaceMetrics:
    detected: bool
    center_x_ratio: float | None = None
    center_y_ratio: float | None = None
    width_ratio: float | None = None
    height_ratio: float | None = None


@dataclass
class Identity:
    id: str
    name: str
    embedding: list[float]


class FaceVerificationService:
    # === UPDATED ===
    MODEL_NAME = "ArcFace"
    DETECTOR_BACKEND = "skip"
    DISTANCE_METRIC = "cosine"
    # === ADDED ===
    INNER_FACE_INDICES = [
        33, 133, 362, 263, 1, 2, 4, 5, 6, 9, 10, 13, 14, 61, 291, 152, 234, 454
    ]

    def __init__(self, identities: list[Identity], match_threshold: float = 0.48):
        self.identities = identities
        self.match_threshold = match_threshold
        # === ADDED ===
        self._embeddings: list[np.ndarray] = []
        self._embedding_size: int | None = None
        for identity in identities:
            vec = np.array(identity.embedding, dtype=np.float64)
            if vec.ndim != 1 or vec.size == 0:
                raise ValueError(f"Identity {identity.id} has an invalid embedding.")
            if self._embedding_size is None:
                self._embedding_size = int(vec.size)
            elif vec.size != self._embedding_size:
                raise ValueError(
                    f"Identity {identity.id} has embedding size {vec.size}, "
                    f"expected {self._embedding_size}."
                )
            self._embeddings.append(vec)
        self._detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        if self._detector.empty():
            raise RuntimeError("OpenCV face detector failed to load.")

        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    # === ADDED ===
    def _extract_landmarks(self, frame: np.ndarray) -> np.ndarray | None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0].landmark
        height, width = frame.shape[:2]
        return np.array(
            [[landmark.x * width, landmark.y * height] for landmark in face_landmarks],
            dtype=np.float32,
        )

    def _eye_aspect_ratio(self, eye_landmarks):
        horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3]) + 1e-6
        ear = (
            np.linalg.norm(eye_landmarks[1] - eye_landmarks[5]) +
            np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        ) / (2 * horizontal)
        return ear

    # === ADDED ===
    def _get_face_box_from_landmarks(self, landmarks: np.ndarray) -> tuple[int, int, int, int]:
        min_xy = landmarks.min(axis=0)
        max_xy = landmarks.max(axis=0)
        x, y = min_xy
        x2, y2 = max_xy
        return int(x), int(y), int(x2 - x), int(y2 - y)

    # === ADDED ===
    def _crop_face(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray | None = None,
        focus: str = "balanced",
    ) -> np.ndarray:
        if landmarks is None:
            landmarks = self._extract_landmarks(frame)

        if landmarks is not None:
            left_eye_center = landmarks[[33, 133]].mean(axis=0)
            right_eye_center = landmarks[[362, 263]].mean(axis=0)
            mouth_center = landmarks[[13, 14]].mean(axis=0)
            nose_tip = landmarks[1]
            face_center = np.mean(
                np.array([left_eye_center, right_eye_center, mouth_center, nose_tip]),
                axis=0,
            )

            dx = right_eye_center[0] - left_eye_center[0]
            dy = right_eye_center[1] - left_eye_center[1]
            angle = np.degrees(np.arctan2(dy, dx))
            rotation_matrix = cv2.getRotationMatrix2D(tuple(face_center), angle, 1.0)
            aligned = cv2.warpAffine(
                frame,
                rotation_matrix,
                (frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

            ones = np.ones((landmarks.shape[0], 1), dtype=np.float32)
            rotated_landmarks = (rotation_matrix @ np.hstack([landmarks, ones]).T).T
            if focus == "inner":
                anchor_points = rotated_landmarks[self.INNER_FACE_INDICES]
                x, y, w, h = self._get_face_box_from_landmarks(anchor_points)
                x_pad = int(w * 0.16)
                y_pad_top = int(h * 0.12)
                y_pad_bottom = int(h * 0.18)
            else:
                anchor_points = rotated_landmarks[self.INNER_FACE_INDICES + [152]]
                x, y, w, h = self._get_face_box_from_landmarks(anchor_points)
                x_pad = int(w * 0.24)
                y_pad_top = int(h * 0.18)
                y_pad_bottom = int(h * 0.22)
            x1 = max(x - x_pad, 0)
            y1 = max(y - y_pad_top, 0)
            x2 = min(x + w + x_pad, aligned.shape[1])
            y2 = min(y + h + y_pad_bottom, aligned.shape[0])
            face = aligned[y1:y2, x1:x2]
            if face.size > 0:
                return face

        face_box = self._detect_largest_face_box(frame)
        if face_box is None:
            raise ValueError("No face detected in the image.")

        x, y, w, h = face_box
        pad = int(max(w, h) * 0.20)
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, frame.shape[1])
        y2 = min(y + h + pad, frame.shape[0])
        return frame[y1:y2, x1:x2]

    # === ADDED ===
    def _represent_face(self, face: np.ndarray) -> np.ndarray:
        result = DeepFace.represent(
            img_path=face,
            model_name=self.MODEL_NAME,
            detector_backend=self.DETECTOR_BACKEND,
            enforce_detection=False,
            normalization="ArcFace",
        )
        if not result:
            raise ValueError("No face detected in the uploaded image.")
        embedding = np.array(result[0]["embedding"], dtype=np.float64)
        embedding /= np.linalg.norm(embedding) + 1e-12
        return embedding

    # === ADDED ===
    def _prepare_face_variants(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        balanced_face = self._crop_face(frame, landmarks=landmarks, focus="balanced")
        inner_face = self._crop_face(frame, landmarks=landmarks, focus="inner")

        equalized_face = cv2.cvtColor(inner_face, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(equalized_face)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        equalized_face = cv2.merge((l_channel, a_channel, b_channel))
        equalized_face = cv2.cvtColor(equalized_face, cv2.COLOR_LAB2BGR)

        return [balanced_face, inner_face, equalized_face]

    # === ADDED ===
    def _liveness_confidence(self, frame: np.ndarray, landmarks: np.ndarray | None = None) -> float:
        if landmarks is None:
            landmarks = self._extract_landmarks(frame)
        if landmarks is None:
            return 0.0

        face = self._crop_face(frame, landmarks)
        if face.size == 0:
            return 0.0

        left_eye = landmarks[[33, 160, 158, 133, 153, 144]]
        right_eye = landmarks[[362, 385, 387, 263, 373, 380]]
        mouth = landmarks[[61, 291, 13, 14]]
        nose_tip = landmarks[1]
        chin = landmarks[152]

        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        eye_balance = 1.0 - min(abs(left_ear - right_ear) / 0.10, 1.0)

        mouth_width = np.linalg.norm(mouth[0] - mouth[1]) + 1e-6
        mouth_open = np.linalg.norm(mouth[2] - mouth[3]) / mouth_width
        face_height = np.linalg.norm(chin - nose_tip) + 1e-6
        eye_span = np.linalg.norm(left_eye[0] - right_eye[3]) + 1e-6
        facial_ratio = eye_span / face_height
        geometry_score = 1.0 - min(abs(facial_ratio - 1.05) / 0.55, 1.0)

        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        texture_score = min(laplacian_var / 180.0, 1.0)

        hsv_face = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        saturation_mean = float(np.mean(hsv_face[:, :, 1])) / 255.0
        saturation_score = min(saturation_mean / 0.35, 1.0)

        # Passive anti-spoof heuristics:
        # - eye openness discourages flat printed faces with closed/blurred eyes
        # - geometry focuses on inner-face landmarks rather than hair/glasses/head coverings
        # - texture helps reject screens or glossy prints
        eye_score = min(max((avg_ear - 0.16) / 0.16, 0.0), 1.0)
        mouth_score = 1.0 - min(abs(mouth_open - 0.10) / 0.18, 1.0)

        score = (
            0.28 * eye_score +
            0.18 * eye_balance +
            0.20 * geometry_score +
            0.24 * texture_score +
            0.10 * saturation_score
        )
        score *= max(mouth_score, 0.55)
        return float(max(0.0, min(score, 1.0)))

    # === UPDATED ===
    def _detect_liveness(self, frame: np.ndarray) -> tuple[str, float]:
        landmarks = self._extract_landmarks(frame)
        score = self._liveness_confidence(frame, landmarks)
        return ("LIVE" if score >= 0.58 else "SPOOF", round(score, 6))

    def assess_frame(self, frame: np.ndarray) -> FrameAssessment:
        metrics = self.analyze_frame(frame)
        if not metrics.detected:
            return FrameAssessment(False, "No face detected")

        if metrics.width_ratio < 0.18:
            return FrameAssessment(False, "Move closer to the camera")
        if metrics.width_ratio > 0.62:
            return FrameAssessment(False, "Move slightly farther away")
        if metrics.center_x_ratio is not None and (metrics.center_x_ratio < 0.22 or metrics.center_x_ratio > 0.78):
            return FrameAssessment(False, "Center your face in the frame")
        if metrics.center_y_ratio is not None and (metrics.center_y_ratio < 0.18 or metrics.center_y_ratio > 0.82):
            return FrameAssessment(False, "Align your face vertically in the frame")

        return FrameAssessment(True, "Face detected and ready")

    def analyze_frame(self, frame: np.ndarray) -> FaceMetrics:
        face_box = self._detect_largest_face_box(frame)
        if face_box is None:
            return FaceMetrics(detected=False)

        frame_height, frame_width = frame.shape[:2]
        x, y, w, h = face_box
        return FaceMetrics(
            detected=True,
            center_x_ratio=(x + w / 2) / frame_width,
            center_y_ratio=(y + h / 2) / frame_height,
            width_ratio=w / frame_width,
            height_ratio=h / frame_height,
        )

    def verify_image(self, frame: np.ndarray) -> VerificationResult:
        liveness, liveness_confidence = self._detect_liveness(frame)
        embedding = self._get_embedding_from_array(frame)
        result = self._find_best_match(embedding)
        result.liveness = liveness
        result.confidence = round(min(result.confidence, liveness_confidence), 6)
        return result

    def verify_image_bytes(self, image_bytes: bytes) -> VerificationResult:
        frame = self.decode_image_bytes(image_bytes)
        return self.verify_image(frame)

    def verify_identity(self, identity_id: str, frame: np.ndarray) -> VerificationResult:
        liveness, liveness_confidence = self._detect_liveness(frame)
        embedding = self._get_embedding_from_array(frame)
        result = self._verify_identity(identity_id, embedding)
        result.liveness = liveness
        result.confidence = round(min(result.confidence, liveness_confidence), 6)
        return result

    def verify_identity_bytes(self, identity_id: str, image_bytes: bytes) -> VerificationResult:
        frame = self.decode_image_bytes(image_bytes)
        return self.verify_identity(identity_id, frame)

    def analyze_image_bytes(self, image_bytes: bytes) -> FaceMetrics:
        frame = self.decode_image_bytes(image_bytes)
        return self.analyze_frame(frame)

    def assess_and_verify_image_bytes(
        self, image_bytes: bytes
    ) -> tuple[FrameAssessment, VerificationResult | None]:
        frame = self.decode_image_bytes(image_bytes)
        assessment = self.assess_frame(frame)
        if not assessment.ready:
            return assessment, None
        return assessment, self.verify_image(frame)

    def decode_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image. Make sure you uploaded a valid image file.")
        return frame

    def _detect_largest_face_box(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(90, 90),
        )

        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        return int(x), int(y), int(w), int(h)

    def _get_embedding_from_path(self, path: Path) -> list[float]:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Could not load database image: {path}")

        # === UPDATED ===
        return self._get_embedding_from_array(frame)

    def _get_embedding_from_array(self, frame: np.ndarray) -> list[float]:
        # === UPDATED ===
        landmarks = self._extract_landmarks(frame)
        embeddings = [
            self._represent_face(face)
            for face in self._prepare_face_variants(frame, landmarks=landmarks)
        ]
        embedding = np.mean(embeddings, axis=0)
        embedding /= np.linalg.norm(embedding) + 1e-12
        if self._embedding_size is not None and embedding.size != self._embedding_size:
            raise ValueError(
                "Embedding size mismatch between uploaded image and stored identity database."
            )
        return embedding.tolist()

    def _find_best_match(self, embedding: list[float]) -> VerificationResult:
        if not self.identities:
            return VerificationResult(matched=False, distance=1.0, liveness="SPOOF", confidence=0.0)
        vec_b = np.array(embedding, dtype=np.float64)
        vec_b /= np.linalg.norm(vec_b) + 1e-12
        best_distance = float('inf')
        best_identity = None
        for i, vec_a in enumerate(self._embeddings):
            vec_a_norm = vec_a / (np.linalg.norm(vec_a) + 1e-12)
            distance = float(1.0 - np.dot(vec_a_norm, vec_b))
            if distance < best_distance:
                best_distance = distance
                best_identity = self.identities[i]
        matched = best_distance <= self.match_threshold
        confidence = max(0.0, 1.0 - best_distance)
        return VerificationResult(
            matched=matched,
            distance=round(best_distance, 6),
            liveness="LIVE",
            confidence=round(confidence, 6),
            identity_id=best_identity.id,
            identity_name=best_identity.name
        )

    def _verify_identity(self, identity_id: str, embedding: list[float]) -> VerificationResult:
        identity = next((id for id in self.identities if id.id == identity_id), None)
        if not identity:
            raise ValueError(f"Identity {identity_id} not found")
        vec_a = np.array(identity.embedding, dtype=np.float64)
        vec_b = np.array(embedding, dtype=np.float64)
        vec_a /= np.linalg.norm(vec_a) + 1e-12
        vec_b /= np.linalg.norm(vec_b) + 1e-12
        distance = float(1.0 - np.dot(vec_a, vec_b))
        matched = distance <= self.match_threshold
        confidence = max(0.0, 1.0 - distance)
        return VerificationResult(
            matched=matched,
            distance=round(distance, 6),
            liveness="LIVE",
            confidence=round(confidence, 6),
            identity_id=identity.id,
            identity_name=identity.name
        )
