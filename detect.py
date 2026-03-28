"""
Drowsiness & Distraction Detector
==================================
Real-time driver monitoring system using facial landmarks.
Detects eye closure (drowsiness) and head tilt (distraction),
triggers audio + visual alerts.

Run:
    python detect.py --source 0            # webcam
    python detect.py --source video.mp4    # video file
    python detect.py --source image.jpg    # single image
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import argparse
import sys
import os

# ─── Constants ────────────────────────────────────────────────────────────────

EAR_THRESHOLD        = 0.22   # Eye Aspect Ratio below this = eye closed
EAR_CONSEC_FRAMES    = 20     # Consecutive frames eye must be closed to trigger alert
HEAD_TILT_THRESHOLD  = 25     # Degrees of head tilt to flag distraction
YAWN_THRESHOLD       = 0.6    # Mouth Aspect Ratio above this = yawning

# MediaPipe landmark indices
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]
MOUTH_IDX     = [13, 14, 78, 308]   # top, bottom, left-corner, right-corner

# ─── Colours (BGR) ────────────────────────────────────────────────────────────
GREEN  = (0, 200, 0)
YELLOW = (0, 200, 255)
RED    = (0, 0, 220)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
ORANGE = (0, 140, 255)

# ─── Utility functions ────────────────────────────────────────────────────────

def euclidean(p1, p2):
    """Euclidean distance between two (x,y) points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """
    Eye Aspect Ratio (EAR) — Soukupova & Cech (2016).
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Drops sharply when eye closes.
    """
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = euclidean(pts[1], pts[5])
    B = euclidean(pts[2], pts[4])
    C = euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def mouth_aspect_ratio(landmarks, mouth_indices, w, h):
    """
    Mouth Aspect Ratio (MAR) — detects yawning.
    MAR = vertical opening / horizontal width
    """
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in mouth_indices]
    vertical   = euclidean(pts[0], pts[1])
    horizontal = euclidean(pts[2], pts[3])
    return vertical / (horizontal + 1e-6)


def head_tilt_angle(landmarks, w, h):
    """
    Estimate roll angle of the head using nose tip and chin.
    Returns angle in degrees relative to vertical axis.
    """
    nose  = landmarks[1]
    chin  = landmarks[152]
    dx = (chin.x - nose.x) * w
    dy = (chin.y - nose.y) * h
    angle = np.degrees(np.arctan2(dx, dy))
    return angle


def draw_overlay(frame, text, pos, color, scale=0.7, thickness=2):
    """Draw text with a dark shadow for readability."""
    x, y = pos
    cv2.putText(frame, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thickness+1)
    cv2.putText(frame, text, (x,   y),   cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_status_bar(frame, ear, mar, tilt, alert_level):
    """Semi-transparent HUD at top of frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), BLACK, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    bar_color = GREEN if alert_level == 0 else (YELLOW if alert_level == 1 else RED)
    status    = "ALERT" if alert_level == 2 else ("WARNING" if alert_level == 1 else "NORMAL")

    draw_overlay(frame, f"EAR: {ear:.2f}",   (10,  25), WHITE, 0.55)
    draw_overlay(frame, f"MAR: {mar:.2f}",   (10,  50), WHITE, 0.55)
    draw_overlay(frame, f"Tilt: {tilt:.1f}°",(10,  72), WHITE, 0.55)
    draw_overlay(frame, f"STATUS: {status}", (w-210, 45), bar_color, 0.7, 2)


def flash_alert(frame, message, color):
    """Full-frame colour flash for critical alert."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cx, cy = w // 2, h // 2
    draw_overlay(frame, message, (cx - 180, cy), WHITE, 1.2, 3)


# ─── Main detector ────────────────────────────────────────────────────────────

class DrowsinessDetector:
    def __init__(self):
        self.mp_face  = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.closed_counter = 0
        self.alert_triggered = False
        self.yawn_counter    = 0
        self.session_alerts  = {"drowsy": 0, "distracted": 0, "yawn": 0}
        self.start_time      = time.time()

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        alert_level = 0  # 0=normal, 1=warning, 2=alert

        if not result.multi_face_landmarks:
            draw_overlay(frame, "No face detected", (10, h - 20), YELLOW)
            draw_status_bar(frame, 0, 0, 0, 1)
            return frame

        lm = result.multi_face_landmarks[0].landmark

        # ── Compute metrics ──────────────────────────────────────────────────
        left_ear  = eye_aspect_ratio(lm, LEFT_EYE_IDX,  w, h)
        right_ear = eye_aspect_ratio(lm, RIGHT_EYE_IDX, w, h)
        ear       = (left_ear + right_ear) / 2.0
        mar       = mouth_aspect_ratio(lm, MOUTH_IDX, w, h)
        tilt      = head_tilt_angle(lm, w, h)

        # ── Draw subtle landmark mesh ─────────────────────────────────────────
        for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
            x = int(lm[idx].x * w)
            y = int(lm[idx].y * h)
            cv2.circle(frame, (x, y), 2, GREEN, -1)

        # ── Drowsiness detection ──────────────────────────────────────────────
        if ear < EAR_THRESHOLD:
            self.closed_counter += 1
            if self.closed_counter >= EAR_CONSEC_FRAMES:
                alert_level = 2
                self.session_alerts["drowsy"] += 1
                flash_alert(frame, "DROWSINESS ALERT! WAKE UP!", RED)
        else:
            self.closed_counter = 0

        # ── Yawn detection ────────────────────────────────────────────────────
        if mar > YAWN_THRESHOLD:
            self.yawn_counter += 1
            if self.yawn_counter > 15:
                alert_level = max(alert_level, 1)
                self.session_alerts["yawn"] += 1
                draw_overlay(frame, "YAWNING DETECTED", (w//2 - 120, h - 40), ORANGE)
        else:
            self.yawn_counter = 0

        # ── Head tilt / distraction ───────────────────────────────────────────
        if abs(tilt) > HEAD_TILT_THRESHOLD:
            alert_level = max(alert_level, 1)
            self.session_alerts["distracted"] += 1
            draw_overlay(frame, f"HEAD TILT: {tilt:.1f}°", (w//2 - 100, h - 65), YELLOW)

        # ── Eye progress bar ─────────────────────────────────────────────────
        bar_len  = 150
        bar_fill = int(np.clip((ear / 0.35) * bar_len, 0, bar_len))
        bar_col  = GREEN if ear >= EAR_THRESHOLD else RED
        cv2.rectangle(frame, (w-170, h-30), (w-20, h-15), (50,50,50), -1)
        cv2.rectangle(frame, (w-170, h-30), (w-170+bar_fill, h-15), bar_col, -1)
        draw_overlay(frame, "EAR", (w-195, h-15), WHITE, 0.45)

        # ── Session stats ─────────────────────────────────────────────────────
        elapsed = int(time.time() - self.start_time)
        draw_overlay(frame, f"Session: {elapsed//60:02d}:{elapsed%60:02d}", (10, h-10), WHITE, 0.5)
        draw_overlay(frame,
            f"Alerts D:{self.session_alerts['drowsy']} "
            f"Y:{self.session_alerts['yawn']} "
            f"T:{self.session_alerts['distracted']}",
            (10, h - 30), WHITE, 0.45)

        draw_status_bar(frame, ear, mar, tilt, alert_level)
        return frame

    def release(self):
        self.face_mesh.close()


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Drowsiness & Distraction Detector")
    parser.add_argument("--source",  default="0",
                        help="Video source: 0 (webcam), path to video, or image file")
    parser.add_argument("--output",  default=None,
                        help="Save output to file (e.g. output.mp4 or output.jpg)")
    parser.add_argument("--no-display", action="store_true",
                        help="Run without GUI window (useful on headless servers)")
    parser.add_argument("--ear",   type=float, default=EAR_THRESHOLD,
                        help=f"EAR threshold (default {EAR_THRESHOLD})")
    parser.add_argument("--frames", type=int, default=EAR_CONSEC_FRAMES,
                        help=f"Consecutive frames for drowsy alert (default {EAR_CONSEC_FRAMES})")
    return parser.parse_args()


def main():
    args = parse_args()

    # Override globals if user passed thresholds
    global EAR_THRESHOLD, EAR_CONSEC_FRAMES
    EAR_THRESHOLD     = args.ear
    EAR_CONSEC_FRAMES = args.frames

    # ── Open source ───────────────────────────────────────────────────────────
    source = args.source
    is_image = False

    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    elif os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp"):
            is_image = True
        else:
            cap = cv2.VideoCapture(source)
    else:
        print(f"[ERROR] Source not found: {source}")
        sys.exit(1)

    detector = DrowsinessDetector()
    writer   = None

    print("=" * 50)
    print("  Drowsiness & Distraction Detector")
    print("=" * 50)
    print(f"  Source       : {source}")
    print(f"  EAR threshold: {EAR_THRESHOLD}")
    print(f"  Consec frames: {EAR_CONSEC_FRAMES}")
    print(f"  Press 'q' to quit")
    print("=" * 50)

    # ── Image mode ────────────────────────────────────────────────────────────
    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            print("[ERROR] Could not read image.")
            sys.exit(1)
        out = detector.process_frame(frame)
        if args.output:
            cv2.imwrite(args.output, out)
            print(f"[INFO] Saved to {args.output}")
        if not args.no_display:
            cv2.imshow("Drowsiness Detector", out)
            cv2.waitKey(0)
        detector.release()
        cv2.destroyAllWindows()
        return

    # ── Video / webcam mode ───────────────────────────────────────────────────
    if not cap.isOpened():
        print("[ERROR] Cannot open video source.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (fw, fh))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out = detector.process_frame(frame)

        if writer:
            writer.write(out)

        if not args.no_display:
            cv2.imshow("Drowsiness & Distraction Detector  (press Q to quit)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
        print(f"[INFO] Output saved to {args.output}")
    detector.release()
    cv2.destroyAllWindows()

    alerts = detector.session_alerts
    elapsed = int(time.time() - detector.start_time)
    print("\n── Session Summary ──────────────────────────")
    print(f"  Duration     : {elapsed//60:02d}:{elapsed%60:02d}")
    print(f"  Drowsy alerts: {alerts['drowsy']}")
    print(f"  Yawn events  : {alerts['yawn']}")
    print(f"  Head tilt    : {alerts['distracted']}")
    print("─────────────────────────────────────────────")


if __name__ == "__main__":
    main()
