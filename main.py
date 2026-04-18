from pathlib import Path
import time

import cv2
import json

from face_service import FaceVerificationService, Identity


BASE_DIR = Path(__file__).resolve().parent
DB_DATA_PATH = BASE_DIR / "database.json"


def main() -> None:
    with open(DB_DATA_PATH, 'r') as f:
        data = json.load(f)
    # === UPDATED ===
    bootstrap_service = FaceVerificationService([])
    identities: list[Identity] = []
    for record in data:
        identity = Identity(id=record['id'], name=record['name'], embedding=record['embedding'])
        image_name = str(record.get("image_name", "")).strip()
        if image_name:
            image_path = BASE_DIR / image_name
            if image_path.exists():
                identity.embedding = bootstrap_service._get_embedding_from_path(image_path)
        identities.append(identity)
    service = FaceVerificationService(identities)
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Camera is not available.")
        return

    print("Press SPACE to capture a face image, or Q to quit.")

    verification_text = ""
    verification_timestamp: float | None = None

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Failed to read a frame from the camera.")
                return

            if verification_text:
                cv2.putText(
                    frame,
                    verification_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0) if verification_text.startswith("Verified") else (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Face Verification", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == 32 and verification_timestamp is None:
                try:
                    result = service.verify_image(frame)
                    if result.matched:
                        verification_text = f"Verified. Distance: {result.distance:.4f}"
                    else:
                        verification_text = f"Not verified. Distance: {result.distance:.4f}"
                    print(verification_text)
                except ValueError as exc:
                    verification_text = str(exc)
                    print(verification_text)

                verification_timestamp = time.time()

            if verification_timestamp is not None:
                elapsed = time.time() - verification_timestamp
                if elapsed >= 2.0:
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
