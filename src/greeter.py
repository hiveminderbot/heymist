#!/usr/bin/env python3
"""heymist-greeter — greets you when you sit down at your laptop.

Polls the webcam for face presence. When a face appears after an absence,
speaks a contextual greeting via heymist TTS.
"""

import logging
import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2

log = logging.getLogger("heymist-greeter")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPEAK_FIFO = Path(os.environ.get(
    "HEYMIST_SPEAK_FIFO",
    str(Path.home() / ".local/share/heymist/speak.fifo"),
))

# How often to check the webcam (seconds)
POLL_INTERVAL = 2

# Minimum seconds away before greeting again
ABSENCE_THRESHOLD = 120  # 2 minutes

# Webcam device
CAMERA_INDEX = int(os.environ.get("HEYMIST_CAMERA", "0"))

# Haar cascade for face detection
# cv2.data.haarcascades doesn't exist in nixpkgs opencv, so search for it
def _find_cascade():
    # Check environment override
    if os.environ.get("HEYMIST_CASCADE"):
        return os.environ["HEYMIST_CASCADE"]
    # Check common locations
    for base in [
        os.path.join(os.path.dirname(cv2.__file__), "data"),
        "/run/current-system/sw/share/opencv4/haarcascades",
    ]:
        p = os.path.join(base, "haarcascade_frontalface_default.xml")
        if os.path.exists(p):
            return p
    # Search nix store for the opencv package
    import glob
    for p in glob.glob("/nix/store/*/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"):
        return p
    raise FileNotFoundError("Could not find haarcascade_frontalface_default.xml")

CASCADE_PATH = _find_cascade()


# ---------------------------------------------------------------------------
# Greetings
# ---------------------------------------------------------------------------

def get_greeting():
    """Generate a contextual greeting based on time of day and randomness."""
    hour = datetime.now().hour

    if hour < 6:
        time_greetings = [
            "Burning the midnight oil? I'm here if you need me.",
            "Late night session. Let's get it done.",
            "Can't sleep either, huh? What are we working on?",
        ]
    elif hour < 12:
        time_greetings = [
            "Good morning! Ready to build something?",
            "Morning. What's on the agenda today?",
            "Hey, good morning. Coffee first, or straight to code?",
        ]
    elif hour < 17:
        time_greetings = [
            "Welcome back. What are we working on?",
            "Hey! Afternoon productivity mode engaged.",
            "Back at it. Where did we leave off?",
        ]
    elif hour < 21:
        time_greetings = [
            "Evening session. What can I help with?",
            "Hey, welcome back. Wrapping things up or starting fresh?",
            "Good evening. Ready when you are.",
        ]
    else:
        time_greetings = [
            "Late one tonight. What are we tackling?",
            "Hey, night owl. I'm here.",
            "Evening. Let's make it count.",
        ]

    return random.choice(time_greetings)


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------

class FaceDetector:
    """Simple webcam face presence detector."""

    def __init__(self):
        self.cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load cascade from {CASCADE_PATH}")

    def check_for_face(self):
        """Capture a frame and check for faces. Returns True if face found."""
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            log.warning("Could not open camera %d", CAMERA_INDEX)
            return False

        try:
            ret, frame = cap.read()
            if not ret:
                return False

            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
            )

            return len(faces) > 0
        finally:
            cap.release()


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

def speak(text):
    """Send text to heymist TTS via FIFO."""
    if not SPEAK_FIFO.exists():
        log.warning("Speak FIFO not found at %s — is heymist running?", SPEAK_FIFO)
        # Fall back to direct piper
        _speak_direct(text)
        return

    try:
        with open(SPEAK_FIFO, "w") as f:
            f.write(text)
    except Exception:
        log.exception("Failed to write to speak FIFO")


def _speak_direct(text):
    """Speak directly via piper if heymist FIFO isn't available."""
    piper_model = Path.home() / ".local/share/piper-voices/en_US-lessac-medium.onnx"
    if not piper_model.exists():
        log.warning("No piper model, can't speak")
        return

    try:
        piper = subprocess.Popen(
            ["piper", "--model", str(piper_model), "--output-raw"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        play = subprocess.Popen(
            ["play", "-t", "raw", "-r", "22050", "-e", "signed", "-b", "16", "-c", "1", "-"],
            stdin=piper.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        piper.stdin.write(text.encode())
        piper.stdin.close()
        play.wait(timeout=30)
        piper.wait(timeout=5)
    except Exception:
        log.exception("Direct TTS error")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    detector = FaceDetector()
    face_present = False
    last_seen = 0.0
    last_greeted = 0.0

    log.info("heymist-greeter started (camera=%d, absence=%ds)",
             CAMERA_INDEX, ABSENCE_THRESHOLD)

    while True:
        try:
            has_face = detector.check_for_face()
            now = time.time()

            if has_face:
                if not face_present:
                    # Face just appeared
                    absence_duration = now - last_seen if last_seen > 0 else float("inf")

                    if absence_duration >= ABSENCE_THRESHOLD:
                        # Been away long enough — greet!
                        greeting = get_greeting()
                        log.info("Face detected after %.0fs absence. Greeting: %s",
                                 absence_duration, greeting)
                        speak(greeting)
                        last_greeted = now
                    else:
                        log.debug("Face returned after %.0fs (below threshold)", absence_duration)

                    face_present = True

                last_seen = now
            else:
                if face_present:
                    log.debug("Face gone")
                    face_present = False

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            log.info("Shutting down")
            break
        except Exception:
            log.exception("Error in main loop")
            time.sleep(5)


if __name__ == "__main__":
    main()
