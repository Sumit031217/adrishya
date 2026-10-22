# ==============================================================================
# ADRISHYA: INTELLIGENT MULTI-OBJECT VIDEO BLURRING FOR DEFENSE SERVICES
# ==============================================================================
# Full-Fledged Architecture Implementation:
#   1. Multi-Source Acquisition (Auto-Probing Webcam [1, 0, 2], Video File, RTSP)
#   2. Hybrid AI Inference (MediaPipe / Haar / YOLOv8 Head + Sensitive Objects)
#   3. Temporal Tracking & Anti-Flicker Persistence Buffer (ByteTrack + Memory)
#   4. Security Policy Decision Engine (Class -> Soft vs. Hard Redaction Tier)
#   5. ROI Extraction with Boundary Clamping & Adaptive Padding
#   6. Dual-Tier Redaction Engine:
#        - Soft Masking: 256-Bit Encrypted ROI Vault + Adaptive Gaussian Blur
#        - Hard Masking: Permanent Pixelation / Solid Opaque Block (Zero Retention)
#   7. Output Generation, Authorized Forensic Restoration & FastAPI Streaming
#
# Interactive Hotkeys (in OpenCV Window):
#   [q] Quit & Save   [m] Cycle Mask Style   [t] Toggle Soft/Hard   [b] Toggle Boxes
# ==============================================================================

import argparse
import base64
import hashlib
import hmac
import json
import os
import platform
import secrets
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import cvzone
import numpy as np
from ultralytics import YOLO

# Optional dependencies with graceful fallback
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

try:
    from cryptography.fernet import Fernet
    HAS_FERNET = True
except ImportError:
    HAS_FERNET = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ------------------------------------------------------------------------------
# STEP 1: CONFIGURATION & SECURITY POLICY DEFINITIONS
# ------------------------------------------------------------------------------

@dataclass
class RedactionRule:
    """Defines how a specific class of sensitive object is redacted."""
    tier: str          # "SOFT" (reversible encrypted) or "HARD" (irreversible)
    style: str         # "GAUSSIAN", "PIXELATE", or "SOLID"
    padding: int = 8   # Extra pixel margin around bounding box


@dataclass
class AdrishyaConfig:
    # Input / Output settings ("auto" probes webcam index 1 first, then 0, then 2)
    source: str = "auto"
    frame_width: int = 640
    frame_height: int = 480
    output_video_path: Optional[str] = "redacted_output.mp4"
    vault_path: str = "recovery_vault.adrishya"
    key_path: str = "secret.key"

    # Model settings
    yolo_model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.50

    # Temporal persistence (prevents flicker during brief occlusion/lighting shifts)
    max_missing_frames: int = 8      # Keep masking for N frames if detection drops
    iou_match_threshold: float = 0.30

    # Target COCO sensitive class IDs:
    # 62: tv/monitor, 63: laptop, 66: keyboard, 67: cell phone, 73: book/document
    target_coco_classes: Dict[int, str] = field(default_factory=lambda: {
        62: "tactical_display",
        63: "laptop_screen",
        66: "input_terminal",
        67: "mobile_device",
        73: "confidential_doc",
    })

    # Security Policy Map: Class Name -> RedactionRule(tier, style, padding)
    policy_rules: Dict[str, RedactionRule] = field(default_factory=lambda: {
        "face":             RedactionRule(tier="SOFT", style="GAUSSIAN", padding=10),
        "tactical_display": RedactionRule(tier="SOFT", style="GAUSSIAN", padding=6),
        "laptop_screen":    RedactionRule(tier="SOFT", style="GAUSSIAN", padding=6),
        "mobile_device":    RedactionRule(tier="HARD", style="PIXELATE", padding=4),
        "confidential_doc": RedactionRule(tier="HARD", style="SOLID",    padding=8),
        "input_terminal":   RedactionRule(tier="HARD", style="PIXELATE", padding=4),
    })

    # Display settings
    window_name: str = "Adrishya | Intelligent Multi-Object Video Redaction"
    show_hud: bool = True
    draw_debug_boxes: bool = True


# ------------------------------------------------------------------------------
# STEP 2: 256-BIT CRYPTOGRAPHIC VAULT FOR REVERSIBLE (SOFT) MASKING
# ------------------------------------------------------------------------------

class StdlibAEADCipher:
    """
    Zero-dependency 256-bit symmetric cipher using SHA-256 CTR stream encryption
    and HMAC-SHA256 authentication (used automatically if 'cryptography' is absent).
    """

    def __init__(self, key_32_bytes: bytes):
        self.enc_key = hashlib.sha256(b"adrishya-enc|" + key_32_bytes).digest()
        self.mac_key = hashlib.sha256(b"adrishya-mac|" + key_32_bytes).digest()

    def _keystream(self, nonce: bytes, length: int) -> bytes:
        blocks = []
        counter = 0
        while len(blocks) * 32 < length:
            blk = hashlib.sha256(self.enc_key + nonce + struct.pack(">I", counter)).digest()
            blocks.append(blk)
            counter += 1
        return b"".join(blocks)[:length]

    def encrypt(self, plaintext: bytes) -> bytes:
        nonce = secrets.token_bytes(16)
        stream = self._keystream(nonce, len(plaintext))
        ciphertext = bytes(a ^ b for a, b in zip(plaintext, stream))
        tag = hmac.new(self.mac_key, nonce + ciphertext, hashlib.sha256).digest()
        return nonce + tag + ciphertext

    def decrypt(self, token: bytes) -> bytes:
        if len(token) < 48:
            raise ValueError("Corrupted encrypted record.")
        nonce, tag, ciphertext = token[:16], token[16:48], token[48:]
        expected_tag = hmac.new(self.mac_key, nonce + ciphertext, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected_tag):
            raise PermissionError("Cryptographic key authentication failed (HMAC mismatch).")
        stream = self._keystream(nonce, len(ciphertext))
        return bytes(a ^ b for a, b in zip(ciphertext, stream))


class CryptoRecoveryVault:
    """
    Encrypts sensitive ROI crops during SOFT masking and restores them during
    authorized forensic review using 256-bit authenticated encryption.
    """

    def __init__(self, key_path: str, vault_path: str, mode: str = "write"):
        self.key_path = key_path
        self.vault_path = vault_path
        self.cipher = self._init_cipher()
        self.file_handle = None

        if mode == "write":
            self.file_handle = open(self.vault_path, "wb")

    def _init_cipher(self):
        if HAS_FERNET:
            if os.path.exists(self.key_path):
                with open(self.key_path, "rb") as f:
                    key = f.read().strip()
                try:
                    return Fernet(key)
                except Exception:
                    pass
            key = Fernet.generate_key()
            with open(self.key_path, "wb") as f:
                f.write(key)
            print(f"[SECURITY] 256-bit Fernet key active: {self.key_path}")
            return Fernet(key)

        # Standard-library 256-bit SHA256-CTR + HMAC-SHA256 cipher
        if os.path.exists(self.key_path):
            with open(self.key_path, "rb") as f:
                raw = f.read().strip()
            key_bytes = hashlib.sha256(raw).digest()
        else:
            raw_hex = secrets.token_hex(32).encode("ascii")
            with open(self.key_path, "wb") as f:
                f.write(raw_hex)
            key_bytes = hashlib.sha256(raw_hex).digest()
            print(f"[SECURITY] Generated new 256-bit cryptographic key: {self.key_path}")

        return StdlibAEADCipher(key_bytes)

    def store_roi(
        self,
        frame_idx: int,
        track_id: int,
        class_name: str,
        bbox: Tuple[int, int, int, int],
        roi_bgr: np.ndarray,
    ) -> None:
        """Compresses, encrypts, and appends an ROI patch to the binary vault."""
        if self.file_handle is None or roi_bgr.size == 0:
            return

        success, encoded_png = cv2.imencode(".png", roi_bgr)
        if not success:
            return

        payload = {
            "frame": frame_idx,
            "track_id": track_id,
            "class": class_name,
            "bbox": list(bbox),
            "png_b64": base64.b64encode(encoded_png.tobytes()).decode("ascii"),
        }
        raw_bytes = json.dumps(payload).encode("utf-8")
        encrypted = self.cipher.encrypt(raw_bytes)

        self.file_handle.write(struct.pack(">I", len(encrypted)))
        self.file_handle.write(encrypted)

    def load_all_records(self) -> Dict[int, List[dict]]:
        """Decrypts all stored ROIs indexed by frame number for video restoration."""
        records_by_frame: Dict[int, List[dict]] = {}
        if not os.path.exists(self.vault_path):
            raise FileNotFoundError(f"Vault file not found: {self.vault_path}")

        with open(self.vault_path, "rb") as f:
            while True:
                len_bytes = f.read(4)
                if not len_bytes:
                    break
                (record_len,) = struct.unpack(">I", len_bytes)
                encrypted = f.read(record_len)
                raw_bytes = self.cipher.decrypt(encrypted)
                record = json.loads(raw_bytes.decode("utf-8"))
                frame_idx = record["frame"]
                records_by_frame.setdefault(frame_idx, []).append(record)

        return records_by_frame

    def close(self) -> None:
        if self.file_handle is not None:
            self.file_handle.flush()
            self.file_handle.close()
            self.file_handle = None


# ------------------------------------------------------------------------------
# STEP 3: TEMPORAL TRACKING & ANTI-FLICKER PERSISTENCE ENGINE
# ------------------------------------------------------------------------------

@dataclass
class TrackedEntity:
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    missing_frames: int = 0


class TemporalPersistenceTracker:
    """
    Maintains object trajectories across consecutive frames so that brief
    detector dropouts (due to motion blur, overlap, or lighting shifts) do not
    expose unredacted sensitive regions.
    """

    def __init__(self, max_missing: int = 8, iou_threshold: float = 0.30):
        self.max_missing = max_missing
        self.iou_threshold = iou_threshold
        self.active_tracks: Dict[int, TrackedEntity] = {}
        self.next_synthetic_id: int = 1000

    @staticmethod
    def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        return inter_area / float(areaA + areaB - inter_area)

    def update(self, detections: List[TrackedEntity]) -> List[TrackedEntity]:
        updated_tracks: Dict[int, TrackedEntity] = {}
        unmatched_prev_ids = set(self.active_tracks.keys())

        for det in detections:
            # 1. Direct ID match from ByteTrack
            if det.track_id > 0 and det.track_id in self.active_tracks:
                prev = self.active_tracks[det.track_id]
                det.bbox = tuple(
                    int(0.70 * d + 0.30 * p) for d, p in zip(det.bbox, prev.bbox)
                )  # type: ignore
                det.missing_frames = 0
                updated_tracks[det.track_id] = det
                unmatched_prev_ids.discard(det.track_id)
                continue

            # 2. IoU spatial match (used for faces or re-identified boxes)
            best_iou = 0.0
            best_id = None
            for prev_id in unmatched_prev_ids:
                prev_track = self.active_tracks[prev_id]
                if prev_track.class_name != det.class_name:
                    continue
                iou = self.compute_iou(det.bbox, prev_track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_id = prev_id

            if best_id is not None and best_iou >= self.iou_threshold:
                prev = self.active_tracks[best_id]
                det.track_id = best_id
                det.bbox = tuple(
                    int(0.75 * d + 0.25 * p) for d, p in zip(det.bbox, prev.bbox)
                )  # type: ignore
                det.missing_frames = 0
                updated_tracks[best_id] = det
                unmatched_prev_ids.discard(best_id)
            else:
                if det.track_id <= 0:
                    det.track_id = self.next_synthetic_id
                    self.next_synthetic_id += 1
                det.missing_frames = 0
                updated_tracks[det.track_id] = det

        # 3. Retain recently disappeared tracks up to max_missing frames
        for prev_id in unmatched_prev_ids:
            ghost = self.active_tracks[prev_id]
            ghost.missing_frames += 1
            if ghost.missing_frames <= self.max_missing:
                updated_tracks[prev_id] = ghost

        self.active_tracks = updated_tracks
        return list(self.active_tracks.values())


# ------------------------------------------------------------------------------
# STEP 4: DUAL-TIER MASK GENERATION (GAUSSIAN, PIXELATION, SOLID BLOCK)
# ------------------------------------------------------------------------------

class RedactionMasker:
    """Applies adaptive Gaussian blur, pixelation, or solid tactical masking."""

    @staticmethod
    def apply_gaussian_blur(roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        k_w = max(15, (w // 3) | 1)
        k_h = max(15, (h // 3) | 1)
        return cv2.GaussianBlur(roi, (k_w, k_h), 30)

    @staticmethod
    def apply_pixelation(roi: np.ndarray, blocks: int = 10) -> np.ndarray:
        h, w = roi.shape[:2]
        x_steps = max(4, min(blocks, max(4, w // 4)))
        y_steps = max(4, min(blocks, max(4, h // 4)))
        small = cv2.resize(roi, (x_steps, y_steps), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def apply_solid_block(roi: np.ndarray, label: str = "REDACTED") -> np.ndarray:
        blocked = np.full_like(roi, (24, 24, 28))
        h, w = blocked.shape[:2]
        cv2.rectangle(blocked, (0, 0), (w - 1, h - 1), (0, 0, 220), 2)
        if w > 70 and h > 28:
            font_scale = min(0.7, w / 180.0)
            cv2.putText(
                blocked,
                label,
                (max(6, w // 8), h // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (200, 200, 255),
                2,
                cv2.LINE_AA,
            )
        return blocked


# ------------------------------------------------------------------------------
# STEP 5: HYBRID AI DETECTOR (MEDIAPIPE / HAAR + YOLOv8 + BYTETRACK)
# ------------------------------------------------------------------------------

class AdrishyaDetector:
    """Combines Face Detection (MediaPipe / Haar / YOLO Person-Head) with YOLOv8 + ByteTrack."""

    def __init__(self, config: AdrishyaConfig):
        self.config = config
        self.device = "cuda:0" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        print(f"[INIT] Loading YOLOv8 model '{config.yolo_model_path}' on {self.device}...")

        self.yolo = YOLO(config.yolo_model_path)

        # Initialize Face Detector (MediaPipe if available, plus Haar Cascade fallback)
        self.mp_face = None
        self.haar_face = None

        if HAS_MEDIAPIPE and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            self.mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=config.confidence_threshold,
            )
            print("[INIT] Face Detector: MediaPipe Full-Range + YOLOv8 Head Localization active.")
        else:
            haar_dir = getattr(cv2.data, "haarcascades", "")
            cascade_path = os.path.join(haar_dir, "haarcascade_frontalface_default.xml")
            if os.path.exists(cascade_path):
                self.haar_face = cv2.CascadeClassifier(cascade_path)
            print("[INIT] Face Detector: OpenCV Haar Cascade + YOLOv8 Head Localization active.")

    def detect_and_track(self, frame: np.ndarray) -> List[TrackedEntity]:
        h, w = frame.shape[:2]
        entities: List[TrackedEntity] = []
        face_boxes: List[Tuple[int, int, int, int]] = []

        # 1. Dedicated Face Localization (MediaPipe or Haar)
        if self.mp_face is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_results = self.mp_face.process(rgb_frame)
            if mp_results.detections:
                for det in mp_results.detections:
                    rel_box = det.location_data.relative_bounding_box
                    x1 = max(0, int(rel_box.xmin * w))
                    y1 = max(0, int(rel_box.ymin * h))
                    x2 = min(w, int((rel_box.xmin + rel_box.width) * w))
                    y2 = min(h, int((rel_box.ymin + rel_box.height) * h))
                    score = float(det.score[0]) if det.score else 0.85
                    fbox = (x1, y1, x2, y2)
                    face_boxes.append(fbox)
                    entities.append(
                        TrackedEntity(track_id=-1, class_name="face", bbox=fbox, confidence=score)
                    )
        elif self.haar_face is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar_face.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5)
            for (fx, fy, fw, fh) in faces:
                fbox = (int(fx), int(fy), int(fx + fw), int(fy + fh))
                face_boxes.append(fbox)
                entities.append(
                    TrackedEntity(track_id=-1, class_name="face", bbox=fbox, confidence=0.82)
                )

        # 2. YOLOv8 + ByteTrack Multi-Object & Person-Head Fallback
        yolo_results = self.yolo.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            device=self.device,
            verbose=False,
        )

        for result in yolo_results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                raw_name = self.yolo.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1

                # If YOLO detects a person (class 0), estimate the head/face sub-region
                # if no dedicated face box already covers the upper body
                if cls_id == 0:
                    pw = x2 - x1
                    ph = y2 - y1
                    head_x1 = x1 + int(pw * 0.18)
                    head_x2 = x2 - int(pw * 0.18)
                    head_y1 = y1
                    head_y2 = y1 + max(30, int(ph * 0.34))
                    head_box = (head_x1, head_y1, head_x2, head_y2)

                    already_covered = any(
                        TemporalPersistenceTracker.compute_iou(head_box, fb) > 0.15
                        for fb in face_boxes
                    )
                    if not already_covered and (head_x2 - head_x1) > 20:
                        entities.append(
                            TrackedEntity(
                                track_id=track_id,
                                class_name="face",
                                bbox=head_box,
                                confidence=conf,
                            )
                        )
                    continue

                if raw_name in self.config.policy_rules:
                    mapped_name = raw_name
                elif cls_id in self.config.target_coco_classes:
                    mapped_name = self.config.target_coco_classes[cls_id]
                else:
                    continue

                entities.append(
                    TrackedEntity(
                        track_id=track_id,
                        class_name=mapped_name,
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                    )
                )

        return entities


# ------------------------------------------------------------------------------
# STEP 6: CORE ADRISHYA PIPELINE (ACQUISITION -> REDACTION -> RESTORATION)
# ------------------------------------------------------------------------------

class AdrishyaPipeline:
    """Orchestrates real-time and offline video redaction and restoration."""

    def __init__(self, config: AdrishyaConfig):
        self.config = config
        self.detector = AdrishyaDetector(config)
        self.tracker = TemporalPersistenceTracker(
            max_missing=config.max_missing_frames,
            iou_threshold=config.iou_match_threshold,
        )
        self.vault = CryptoRecoveryVault(
            key_path=config.key_path,
            vault_path=config.vault_path,
            mode="write",
        )
        self.fps_ema = 0.0
        self.override_style: Optional[str] = None
        self.override_tier: Optional[str] = None

    def open_video_source(self, source: str) -> cv2.VideoCapture:
        """
        Robust cross-platform camera / file / RTSP opener.
        Probes camera indices [1, 0, 2] across CAP_DSHOW, CAP_MSMF, and CAP_ANY
        on Windows and verifies actual frame capture before returning.
        """
        if source.lower() == "auto" or source.isdigit():
            if source.lower() == "auto":
                candidate_indices = [1, 0, 2]
            else:
                req_idx = int(source)
                candidate_indices = [req_idx] + [i for i in (1, 0, 2) if i != req_idx]

            if platform.system() == "Windows":
                backends = [
                    ("CAP_DSHOW", cv2.CAP_DSHOW),
                    ("CAP_MSMF", cv2.CAP_MSMF),
                    ("CAP_ANY", cv2.CAP_ANY),
                ]
            else:
                backends = [("CAP_ANY", cv2.CAP_ANY)]

            for idx in candidate_indices:
                for backend_name, backend_flag in backends:
                    cap = (
                        cv2.VideoCapture(idx)
                        if backend_flag == cv2.CAP_ANY
                        else cv2.VideoCapture(idx, backend_flag)
                    )
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
                        ok, test_frame = cap.read()
                        if ok and test_frame is not None and test_frame.size > 0:
                            print(
                                f"[CAMERA] Connected to webcam index {idx} via {backend_name} "
                                f"({test_frame.shape[1]}x{test_frame.shape[0]})."
                            )
                            return cap
                        cap.release()

            raise RuntimeError(
                "Could not open any webcam on indices [1, 0, 2]. "
                "Check that your camera is plugged in and not locked by another app, "
                "or pass a video file via: python main.py --source path/to/video.mp4"
            )

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video file or stream: {source}")
        return cap

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, dict]:
        """Runs the 7-step Adrishya methodology on a single video frame."""
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        # Step 2 & 3: AI Inference + Temporal Tracking
        raw_detections = self.detector.detect_and_track(frame)
        tracked_objects = self.tracker.update(raw_detections)

        soft_count = 0
        hard_count = 0

        # Step 4, 5, 6: Policy Evaluation, ROI Extraction & Dual-Tier Masking
        for obj in tracked_objects:
            base_rule = self.config.policy_rules.get(
                obj.class_name,
                RedactionRule(tier="HARD", style="SOLID", padding=6),
            )
            active_tier = self.override_tier or base_rule.tier
            active_style = self.override_style or base_rule.style

            pad = base_rule.padding
            x1 = max(0, obj.bbox[0] - pad)
            y1 = max(0, obj.bbox[1] - pad)
            x2 = min(w, obj.bbox[2] + pad)
            y2 = min(h, obj.bbox[3] + pad)

            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2].copy()

            if active_tier == "SOFT":
                soft_count += 1
                self.vault.store_roi(
                    frame_idx=frame_idx,
                    track_id=obj.track_id,
                    class_name=obj.class_name,
                    bbox=(x1, y1, x2, y2),
                    roi_bgr=roi,
                )
            else:
                hard_count += 1
                del roi

            # Apply selected visual mask
            if active_style == "GAUSSIAN":
                frame[y1:y2, x1:x2] = RedactionMasker.apply_gaussian_blur(frame[y1:y2, x1:x2])
            elif active_style == "PIXELATE":
                frame[y1:y2, x1:x2] = RedactionMasker.apply_pixelation(frame[y1:y2, x1:x2])
            else:
                frame[y1:y2, x1:x2] = RedactionMasker.apply_solid_block(
                    frame[y1:y2, x1:x2], label=f"{active_tier}:{obj.class_name.upper()}"
                )

            if self.config.draw_debug_boxes:
                box_color = (255, 140, 0) if active_tier == "SOFT" else (0, 0, 255)
                cvzone.cornerRect(
                    frame,
                    (x1, y1, x2 - x1, y2 - y1),
                    l=20,
                    t=4,
                    rt=1,
                    colorR=box_color,
                    colorC=(0, 255, 0),
                )
                tag = f"ID:{obj.track_id} {obj.class_name} [{active_tier}|{active_style[:4]}]"
                cvzone.putTextRect(
                    frame,
                    tag,
                    (x1, max(30, y1 - 6)),
                    scale=0.95,
                    thickness=1,
                    colorT=(255, 255, 255),
                    colorR=box_color,
                    font=cv2.FONT_HERSHEY_PLAIN,
                    offset=5,
                )

        # Smooth FPS calculation (Exponential Moving Average)
        dt = max(1e-6, time.perf_counter() - t0)
        inst_fps = 1.0 / dt
        self.fps_ema = inst_fps if self.fps_ema == 0.0 else (0.88 * self.fps_ema + 0.12 * inst_fps)

        if self.config.show_hud:
            cvzone.putTextRect(
                frame,
                f"FPS: {self.fps_ema:.1f} | Soft: {soft_count} | Hard: {hard_count}",
                (16, 34),
                scale=1.1,
                thickness=2,
                colorT=(255, 255, 255),
                colorR=(20, 20, 20),
                offset=7,
            )
            cvzone.putTextRect(
                frame,
                "Keys: [Q] Quit  [M] Mask Style  [T] Soft/Hard Tier  [B] Boxes",
                (16, h - 16),
                scale=0.85,
                thickness=1,
                colorT=(220, 255, 220),
                colorR=(25, 25, 25),
                offset=5,
            )

        stats = {
            "frame": frame_idx,
            "fps": round(self.fps_ema, 1),
            "soft_redactions": soft_count,
            "hard_redactions": hard_count,
        }
        return frame, stats

    def run(self, headless: bool = False) -> None:
        try:
            cap = self.open_video_source(self.config.source)
        except Exception:
            self.vault.close()
            raise

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.config.frame_width)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.config.frame_height)
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        out_fps = src_fps if (src_fps and 1.0 < src_fps <= 120.0) else 25.0

        writer = None
        if self.config.output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.config.output_video_path, fourcc, out_fps, (width, height))

        styles = [None, "GAUSSIAN", "PIXELATE", "SOLID"]
        tiers = [None, "SOFT", "HARD"]
        style_idx = 0
        tier_idx = 0

        frame_idx = 0
        print("[START] Adrishya Redaction Loop running.")
        print("        Hotkeys inside video window: [q] Quit | [m] Mask Style | [t] Tier | [b] Boxes")

        try:
            while True:
                success, frame = cap.read()
                if not success or frame is None:
                    print("[INFO] End of video stream reached.")
                    break

                redacted_frame, _ = self.process_frame(frame, frame_idx)
                frame_idx += 1

                if writer is not None:
                    writer.write(redacted_frame)

                if not headless:
                    cv2.imshow(self.config.window_name, redacted_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("[STOP] 'q' pressed by operator.")
                        break
                    elif key == ord("m"):
                        style_idx = (style_idx + 1) % len(styles)
                        self.override_style = styles[style_idx]
                        print(f"[MODE] Mask style override: {self.override_style or 'POLICY DEFAULT'}")
                    elif key == ord("t"):
                        tier_idx = (tier_idx + 1) % len(tiers)
                        self.override_tier = tiers[tier_idx]
                        print(f"[MODE] Security tier override: {self.override_tier or 'POLICY DEFAULT'}")
                    elif key == ord("b"):
                        self.config.draw_debug_boxes = not self.config.draw_debug_boxes
                        print(f"[MODE] Debug bounding boxes: {self.config.draw_debug_boxes}")

        except KeyboardInterrupt:
            print("[STOP] Interrupted by operator.")
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            self.vault.close()
            if not headless:
                cv2.destroyAllWindows()
            print(f"[DONE] Processed {frame_idx} frames.")
            if self.config.output_video_path:
                print(f"       Redacted video saved to : {self.config.output_video_path}")
            print(f"       Encrypted vault saved to: {self.config.vault_path}")


# ------------------------------------------------------------------------------
# STEP 7: AUTHORIZED FORENSIC RESTORATION UTILITY
# ------------------------------------------------------------------------------

def restore_redacted_video(
    redacted_video_path: str,
    vault_path: str,
    key_path: str,
    restored_output_path: str,
) -> None:
    """
    Decrypts SOFT-masked regions using the secret key and overlays the original
    pixel patches back onto the redacted video for authorized forensic review.
    HARD-masked regions remain permanently redacted.
    """
    vault = CryptoRecoveryVault(key_path=key_path, vault_path=vault_path, mode="read")
    records_by_frame = vault.load_all_records()

    cap = cv2.VideoCapture(redacted_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open redacted video: {redacted_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = cv2.VideoWriter(
        restored_output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        for rec in records_by_frame.get(frame_idx, []):
            x1, y1, x2, y2 = rec["bbox"]
            png_bytes = base64.b64decode(rec["png_b64"])
            roi_arr = np.frombuffer(png_bytes, dtype=np.uint8)
            restored_roi = cv2.imdecode(roi_arr, cv2.IMREAD_COLOR)
            if restored_roi is not None and restored_roi.shape[:2] == (y2 - y1, x2 - x1):
                frame[y1:y2, x1:x2] = restored_roi

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[RESTORED] Forensic restoration complete: {restored_output_path} ({frame_idx} frames)")


# ------------------------------------------------------------------------------
# STEP 8: FASTAPI BACKEND SERVER (LIVE MJPEG STREAM & POLICY API)
# ------------------------------------------------------------------------------

def create_fastapi_app(config: AdrishyaConfig):
    """Creates the FastAPI service specified in the Adrishya technology stack."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, StreamingResponse
    from pydantic import BaseModel

    app = FastAPI(title="Adrishya Privacy-Preserving Defense API", version="2.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    pipeline = AdrishyaPipeline(config)

    @app.get("/", response_class=HTMLResponse)
    def serve_dashboard():
        if os.path.exists("adrishya_dashboard.html"):
            with open("adrishya_dashboard.html", "r", encoding="utf-8") as f:
                return f.read()
        return "<h2>Adrishya API Online</h2><p>Place <code>adrishya_dashboard.html</code> in this folder or open <a href='/stream'>/stream</a>.</p>"

    class PolicyUpdateRequest(BaseModel):
        class_name: str
        tier: str   # "SOFT" or "HARD"
        style: str  # "GAUSSIAN", "PIXELATE", or "SOLID"
        padding: int = 8

    @app.get("/health")
    def health_check():
        return {"status": "online", "device": pipeline.detector.device}

    @app.get("/policies")
    def get_policies():
        return {k: vars(v) for k, v in pipeline.config.policy_rules.items()}

    @app.post("/policies")
    def update_policy(req: PolicyUpdateRequest):
        if req.tier not in {"SOFT", "HARD"} or req.style not in {"GAUSSIAN", "PIXELATE", "SOLID"}:
            raise HTTPException(status_code=400, detail="Invalid tier or style.")
        pipeline.config.policy_rules[req.class_name] = RedactionRule(
            tier=req.tier, style=req.style, padding=req.padding
        )
        return {"updated": req.class_name, "rule": vars(pipeline.config.policy_rules[req.class_name])}

    def mjpeg_generator():
        cap = pipeline.open_video_source(pipeline.config.source)
        frame_idx = 0
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                redacted, _ = pipeline.process_frame(frame, frame_idx)
                frame_idx += 1
                ok_jpg, buf = cv2.imencode(".jpg", redacted)
                if ok_jpg:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                    )
        finally:
            cap.release()
            pipeline.vault.close()

    @app.get("/stream")
    def video_stream():
        return StreamingResponse(
            mjpeg_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return app


# ------------------------------------------------------------------------------
# CLI ENTRY POINT
# ------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Adrishya Multi-Object Video Blurring System")
    parser.add_argument("--mode", choices=["redact", "restore", "api"], default="redact")
    parser.add_argument(
        "--source",
        default="auto",
        help="Camera index ('auto', '1', '0'), video file path, or RTSP URL",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLOv8 weights (.pt or .engine)")
    parser.add_argument("--output", default="redacted_output.mp4", help="Output video file path")
    parser.add_argument("--vault", default="recovery_vault.adrishya", help="Encrypted ROI vault path")
    parser.add_argument("--key", default="secret.key", help="Cryptographic key file path")
    parser.add_argument("--restored-output", default="restored_forensic.mp4", help="Restored video path")
    parser.add_argument("--headless", action="store_true", help="Run without OpenCV GUI window")
    parser.add_argument("--no-boxes", action="store_true", help="Hide bounding box overlays on output")
    parser.add_argument("--port", type=int, default=8000, help="Port for FastAPI server mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = AdrishyaConfig(
        source=args.source,
        yolo_model_path=args.model,
        output_video_path=args.output,
        vault_path=args.vault,
        key_path=args.key,
        draw_debug_boxes=not args.no_boxes,
    )

    if args.mode == "redact":
        engine = AdrishyaPipeline(cfg)
        engine.run(headless=args.headless)
    elif args.mode == "restore":
        restore_redacted_video(
            redacted_video_path=args.output,
            vault_path=args.vault,
            key_path=args.key,
            restored_output_path=args.restored_output,
        )
    elif args.mode == "api":
        import uvicorn
        fastapi_app = create_fastapi_app(cfg)
        uvicorn.run(fastapi_app, host="0.0.0.0", port=args.port)