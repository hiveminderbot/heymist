#!/usr/bin/env python3
"""heymist calibration tool.

Records ambient noise and wake phrase samples to find optimal
VAD thresholds for your voice and environment.
"""

import numpy as np
import sounddevice as sd
import webrtcvad
import yaml
import time
import sys
from pathlib import Path

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)
CONFIG_PATH = Path.home() / ".config/heymist/config.yaml"
NUM_SAMPLES = 5


def record_frames(duration_s):
    """Record audio and return per-frame energy levels."""
    frames = int(duration_s * 1000 / FRAME_MS)
    energies = []
    vad = webrtcvad.Vad(3)
    vad_hits = 0

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME_SIZE
    ) as stream:
        for _ in range(frames):
            frame, _ = stream.read(FRAME_SIZE)
            frame_bytes = frame.tobytes()
            samples = np.frombuffer(frame_bytes, dtype=np.int16)
            energy = np.sqrt(np.mean(samples.astype(float) ** 2)) / 32768.0
            energies.append(energy)
            try:
                if vad.is_speech(frame_bytes, SAMPLE_RATE):
                    vad_hits += 1
            except Exception:
                pass

    return energies, vad_hits


def main():
    print("=" * 50)
    print("  heymist calibration")
    print("=" * 50)
    print()

    # Step 1: measure ambient noise
    print("Step 1: Measuring ambient noise...")
    print("  Stay quiet for 5 seconds.")
    print("  (Don't talk, just let it hear your room)")
    print()
    time.sleep(1)
    print("  Recording...", flush=True)
    ambient_energies, ambient_vad = record_frames(5)
    ambient_mean = np.mean(ambient_energies)
    ambient_p95 = np.percentile(ambient_energies, 95)
    ambient_max = np.max(ambient_energies)
    print(f"  Done. Noise floor: mean={ambient_mean:.4f}, p95={ambient_p95:.4f}, max={ambient_max:.4f}")
    print(f"  VAD triggered on {ambient_vad} frames (should be low)")
    print()

    # Step 2: record wake phrase samples
    speech_energies_all = []
    speech_min_energies = []

    print(f"Step 2: Say 'hey mist' {NUM_SAMPLES} times.")
    print("  Speak at your normal volume and distance from the mic.")
    print()

    for i in range(NUM_SAMPLES):
        print(f"  Sample {i+1}/{NUM_SAMPLES}: Say 'hey mist' now...", flush=True)
        energies, vad_hits = record_frames(3)

        # find the speech portion (frames above ambient)
        threshold = ambient_p95 * 2
        speech_frames = [e for e in energies if e > threshold]

        if speech_frames:
            speech_min = np.min(speech_frames)
            speech_mean = np.mean(speech_frames)
            speech_max = np.max(speech_frames)
            speech_energies_all.extend(speech_frames)
            speech_min_energies.append(speech_min)
            print(f"    Detected speech: min={speech_min:.4f}, mean={speech_mean:.4f}, max={speech_max:.4f}")
        else:
            print(f"    No speech detected! Try speaking louder or closer to mic.")

        time.sleep(0.5)

    if not speech_min_energies:
        print("\nNo speech detected in any sample. Check your mic.")
        sys.exit(1)

    # Step 3: calculate optimal threshold
    print()
    print("=" * 50)
    print("  Results")
    print("=" * 50)
    print()

    noise_ceiling = ambient_p95
    speech_floor = np.min(speech_min_energies)
    speech_mean = np.mean(speech_energies_all)

    # threshold should be between noise ceiling and speech floor
    # use geometric mean for a good midpoint on the log scale
    optimal = np.sqrt(noise_ceiling * speech_floor)

    # clamp to reasonable range
    optimal = max(optimal, 0.005)
    optimal = min(optimal, 0.1)

    print(f"  Ambient noise ceiling (p95):  {noise_ceiling:.4f}")
    print(f"  Speech floor (quietest word): {speech_floor:.4f}")
    print(f"  Speech mean energy:           {speech_mean:.4f}")
    print(f"  Separation ratio:             {speech_floor / noise_ceiling:.1f}x")
    print()
    print(f"  Recommended threshold:        {optimal:.4f}")
    print()

    # Step 4: offer to update config
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f) or {}
        current = config.get("audio", {}).get("vad_threshold", "not set")
        print(f"  Current threshold in config:  {current}")
        print()

        answer = input(f"  Update config to {optimal:.4f}? [Y/n] ").strip().lower()
        if answer in ("", "y", "yes"):
            with open(CONFIG_PATH) as f:
                raw = f.read()

            # replace the threshold line
            import re
            new_raw = re.sub(
                r'(vad_threshold:\s*)[\d.]+',
                f'\\g<1>{optimal:.4f}',
                raw,
            )
            with open(CONFIG_PATH, "w") as f:
                f.write(new_raw)

            print(f"  Updated! Restart heymist: systemctl --user restart heymist")
        else:
            print(f"  Skipped. To apply manually, set vad_threshold: {optimal:.4f}")
    else:
        print(f"  Config not found at {CONFIG_PATH}")
        print(f"  Set vad_threshold: {optimal:.4f}")


if __name__ == "__main__":
    main()
