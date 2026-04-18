from pathlib import Path

import hashlib
import re
import threading
from collections import defaultdict

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import json
import os
import time

from face_service import FaceVerificationService, Identity


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
DB_DATA_PATH = BASE_DIR / "database.json"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
app = FastAPI(title="Secure Face Verification")
app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")
_service_cache: FaceVerificationService | None = None
_service_cache_mtime: float | None = None
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
IDENTITY_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,64}$")


class RateLimiter:
    # === UPDATED ===
    def __init__(self, max_requests=10, window=60):
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, ip):
        with self._lock:
            now = time.time()
            self.requests[ip] = [t for t in self.requests[ip] if now - t < self.window]
            if len(self.requests[ip]) >= self.max_requests:
                return False
            self.requests[ip].append(now)
            return True


# === ADDED ===
class ReplayProtector:
    def __init__(self, ttl_seconds: int = 120):
        self.ttl_seconds = ttl_seconds
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def register(self, client_key: str, payload: bytes) -> bool:
        now = time.time()
        fingerprint = hashlib.sha256(payload).hexdigest()
        key = f"{client_key}:{fingerprint}"
        with self._lock:
            self._seen = {
                saved_key: ts
                for saved_key, ts in self._seen.items()
                if now - ts < self.ttl_seconds
            }
            if key in self._seen:
                return False
            self._seen[key] = now
            return True


limiter = RateLimiter()
replay_protector = ReplayProtector()


# === ADDED ===
def _validate_identity_record(record: dict, index: int) -> Identity:
    if "embedding" not in record or not isinstance(record["embedding"], list):
        raise ValueError(f"Identity record #{index} must store an embedding list.")

    if any(field in record for field in ("image", "image_path", "raw_image", "face_image")):
        raise ValueError(
            f"Identity record #{index} must store facial embeddings instead of raw images."
        )

    identity_id = str(record.get("id", "")).strip()
    if not IDENTITY_ID_PATTERN.fullmatch(identity_id):
        raise ValueError(f"Identity record #{index} has an invalid id.")

    name = str(record.get("name", "")).strip()
    if not name:
        raise ValueError(f"Identity record #{index} is missing a valid name.")

    embedding = []
    for value in record["embedding"]:
        if not isinstance(value, (int, float)):
            raise ValueError(f"Identity record #{index} has a non-numeric embedding value.")
        embedding.append(float(value))

    return Identity(id=identity_id, name=name, embedding=embedding)


# === ADDED ===
async def _read_validated_image(image: UploadFile) -> bytes:
    if not image.content_type or image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPEG, PNG, or WEBP face image.",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large")
    return image_bytes


# === ADDED ===
def _client_identifier(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# === ADDED ===
def _enforce_request_security(request: Request, image_bytes: bytes) -> None:
    client_id = _client_identifier(request)
    if not limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    if not replay_protector.register(client_id, image_bytes):
        raise HTTPException(status_code=409, detail="Replay attack detected")


# === ADDED ===
def _format_verification_response(result) -> dict[str, float | bool | str]:
    return {
        "match": result.matched,
        "similarity_score": round(1 - result.distance, 6),
        "liveness": result.liveness,
        "confidence": result.confidence,
        "verified": result.matched and result.liveness == "LIVE",
    }


# === ADDED ===
def _validate_identity_id(identity_id: str) -> str:
    cleaned = identity_id.strip()
    if not IDENTITY_ID_PATTERN.fullmatch(cleaned):
        raise HTTPException(status_code=400, detail="Invalid identity id format")
    return cleaned


def get_service() -> FaceVerificationService:
    global _service_cache, _service_cache_mtime

    if not DB_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Database file not found: {DB_DATA_PATH}. "
            "Please add a 'database.json' file with identity data."
        )

    current_mtime = os.path.getmtime(DB_DATA_PATH)
    if _service_cache is None or _service_cache_mtime != current_mtime:
        with open(DB_DATA_PATH, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("database.json must contain a list of identities.")
        # === ADDED ===
        embedding_service = FaceVerificationService([])
        identities: list[Identity] = []
        for index, record in enumerate(data, start=1):
            identity = _validate_identity_record(record, index)
            image_name = str(record.get("image_name", "")).strip()
            if image_name:
                image_path = BASE_DIR / image_name
                if image_path.exists():
                    identity.embedding = embedding_service._get_embedding_from_path(image_path)
            identities.append(identity)
        _service_cache = FaceVerificationService(identities)
        _service_cache_mtime = current_mtime

    return _service_cache


@app.get("/")
def home() -> FileResponse:
    return FileResponse(
        WEB_DIR / "index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze-frame")
async def analyze_frame(image: UploadFile = File(...)) -> dict[str, float | bool]:
    try:
        image_bytes = await _read_validated_image(image)
        service = get_service()
        metrics = service.analyze_image_bytes(image_bytes)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Frame analysis failed: {exc}") from exc

    return {
        "detected": metrics.detected,
        "center_x_ratio": metrics.center_x_ratio,
        "center_y_ratio": metrics.center_y_ratio,
        "width_ratio": metrics.width_ratio,
        "height_ratio": metrics.height_ratio,
    }


@app.post("/verify-frame")
async def verify_frame(request: Request, image: UploadFile = File(...)) -> dict[str, float | bool | str | None]:
    try:
        image_bytes = await _read_validated_image(image)
        _enforce_request_security(request, image_bytes)
        service = get_service()
        assessment, result = service.assess_and_verify_image_bytes(image_bytes)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Verification failed: {exc}") from exc

    if result is None:
        return {
            "status": "waiting",
            "message": assessment.message,
            "matched": None,
            "distance": None,
            "threshold": service.match_threshold,
        }

    return {
        "status": "frame_checked",
        "message": assessment.message,
        "matched": result.matched,
        "distance": result.distance,
        "threshold": service.match_threshold,
        "liveness": result.liveness,
        "confidence": result.confidence,
        "verified": result.matched and result.liveness == "LIVE",
        "identity_id": result.identity_id,
        "identity_name": result.identity_name
    }


@app.post("/face-verify")
async def face_verify(request: Request, image: UploadFile = File(...)) -> dict[str, float | bool | str | None]:
    try:
        image_bytes = await _read_validated_image(image)
        _enforce_request_security(request, image_bytes)
        service = get_service()
        result = service.verify_image_bytes(image_bytes)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Verification failed: {exc}") from exc

    return _format_verification_response(result)


@app.post("/verify-identity/{identity_id}")
async def verify_identity(identity_id: str, request: Request, image: UploadFile = File(...)) -> dict[str, float | bool | str | None]:
    try:
        identity_id = _validate_identity_id(identity_id)
        image_bytes = await _read_validated_image(image)
        _enforce_request_security(request, image_bytes)
        service = get_service()
        result = service.verify_identity_bytes(identity_id, image_bytes)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Verification failed: {exc}") from exc

    return _format_verification_response(result)


# === ADDED ===
@app.post("/verify-identity")
async def verify_identity_form(
    request: Request,
    identity_id: str = Form(...),
    image: UploadFile = File(...),
) -> dict[str, float | bool | str | None]:
    return await verify_identity(identity_id, request, image)


@app.post("/verify")
async def verify(request: Request, image: UploadFile = File(...)) -> dict[str, float | bool | str | None]:
    return await face_verify(request, image)
