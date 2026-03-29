#!/usr/bin/env python3
"""heymist - voice input daemon with neural wake word detection.

Uses openwakeword for lightweight always-on wake phrase detection,
then whisper.cpp for transcription of the actual command.
"""

import collections
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import webrtcvad
import yaml

log = logging.getLogger("heymist")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "wake_phrase": "hey mist",
    "backend": "whisper-local",
    "whisper": {"model": "small.en", "threads": 4},
    "cohere": {"api_key": ""},
    "audio": {
        "sample_rate": 16000,
        "vad_threshold": 0.015,
        "silence_duration": 2.0,
        "min_speech": 0.5,
        "max_duration": 30,
    },
    "wakeword": {
        # "builtin:<name>" for pre-trained, or path to .tflite/.onnx
        "model": "builtin:hey_jarvis",
        "threshold": 0.5,
    },
    "output": "ydotool",
    "prefix": "[voice] ",
    "feedback": {"enabled": True, "type": "notify"},
}


def load_config():
    config = dict(DEFAULT_CONFIG)
    config_path = Path(os.environ.get(
        "HEYMIST_CONFIG", Path.home() / ".config/heymist/config.yaml"
    ))
    if config_path.exists():
        with open(config_path) as f:
            user = yaml.safe_load(f) or {}
        _deep_merge(config, user)
    return config


def _deep_merge(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ---------------------------------------------------------------------------
# Wake word detection (openwakeword)
# ---------------------------------------------------------------------------

OWW_CHUNK = 1280  # 80ms at 16kHz — required by openwakeword


class WakeWordDetector:
    """Neural wake word detection using openwakeword."""

    def __init__(self, config):
        from pyopen_wakeword import OpenWakeWord, OpenWakeWordFeatures, Model

        self.threshold = config["wakeword"]["threshold"]
        self.sample_rate = config["audio"]["sample_rate"]
        model_spec = config["wakeword"]["model"]

        # Load feature extractor (shared across all models)
        self.features = OpenWakeWordFeatures.from_builtin()

        # Load wake word model
        if model_spec.startswith("builtin:"):
            name = model_spec.split(":", 1)[1]
            builtin_model = Model(name)
            self.oww = OpenWakeWord.from_builtin(builtin_model)
            log.info("Loaded builtin wake word model: %s", name)
        else:
            model_path = Path(model_spec).expanduser()
            if not model_path.exists():
                log.error("Wake word model not found: %s", model_path)
                sys.exit(1)
            self.oww = OpenWakeWord.from_model(str(model_path))
            log.info("Loaded custom wake word model: %s", model_path)

    def listen_for_wake_word(self):
        """Block until wake word is detected."""
        log.info("Listening for wake word...")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=OWW_CHUNK,
        ) as stream:
            while True:
                # Skip frames while TTS is playing
                if tts_active.is_set():
                    stream.read(OWW_CHUNK)
                    continue

                frame, _ = stream.read(OWW_CHUNK)
                frame_bytes = frame.tobytes()

                # Extract embeddings from audio
                embeddings = list(self.features.process_streaming(frame_bytes))

                # Score each embedding against the wake word model
                for emb in embeddings:
                    scores = list(self.oww.process_streaming(emb))
                    for score in scores:
                        if score >= self.threshold:
                            log.info("Wake word detected! (score=%.3f)", score)
                            self.oww.reset()
                            return

    def reset(self):
        """Reset detector state after a detection."""
        self.oww.reset()


# ---------------------------------------------------------------------------
# Audio capture + VAD (for recording commands after wake word)
# ---------------------------------------------------------------------------

FRAME_MS = 30  # webrtcvad frame size in ms


class CommandRecorder:
    """Records speech after wake word detection until silence."""

    def __init__(self, config):
        self.sample_rate = config["audio"]["sample_rate"]
        self.frame_size = int(self.sample_rate * FRAME_MS / 1000)
        self.silence_frames = int(
            config["audio"]["silence_duration"] * 1000 / FRAME_MS
        )
        self.min_speech_frames = int(
            config["audio"]["min_speech"] * 1000 / FRAME_MS
        )
        self.max_frames = int(
            config["audio"]["max_duration"] * 1000 / FRAME_MS
        )
        self.energy_threshold = config["audio"]["vad_threshold"]
        self.vad = webrtcvad.Vad(2)  # less aggressive for command recording

    def record_command(self):
        """Record audio until silence, return raw audio bytes."""
        speech_frames = []
        silent_count = 0
        has_speech = False

        log.info("Recording command...")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_size,
        ) as stream:
            while True:
                frame, _ = stream.read(self.frame_size)
                frame_bytes = frame.tobytes()
                speech_frames.append(frame_bytes)

                is_speech = self._is_speech(frame_bytes)

                if is_speech:
                    has_speech = True
                    silent_count = 0
                else:
                    silent_count += 1

                # Wait for speech to start (grace period of 3s)
                if not has_speech and len(speech_frames) > int(3000 / FRAME_MS):
                    log.info("No speech after wake word, resetting")
                    return None

                # Stop after sufficient silence (only if we've heard speech)
                if has_speech and silent_count >= self.silence_frames:
                    if len(speech_frames) >= self.min_speech_frames:
                        log.debug("Command ended (%d frames)", len(speech_frames))
                        return b"".join(speech_frames)
                    else:
                        log.debug("Too short, resetting")
                        return None

                # Max duration safety
                if len(speech_frames) >= self.max_frames:
                    log.debug("Max duration reached")
                    return b"".join(speech_frames)

    def _is_speech(self, frame_bytes):
        """Check if frame contains speech."""
        try:
            vad_result = self.vad.is_speech(frame_bytes, self.sample_rate)
        except Exception:
            vad_result = False

        # Energy check
        samples = np.frombuffer(frame_bytes, dtype=np.int16)
        energy = np.sqrt(np.mean(samples.astype(float) ** 2)) / 32768.0
        energy_result = energy > self.energy_threshold

        return vad_result and energy_result


# ---------------------------------------------------------------------------
# Transcription backends
# ---------------------------------------------------------------------------


def transcribe_whisper_local(audio_bytes, config):
    """Transcribe using whisper.cpp."""
    whisper_bin = shutil.which("whisper-cli")
    if not whisper_bin:
        log.error("whisper-cli not found in PATH")
        return ""

    model_name = config["whisper"]["model"]
    model_dir = Path.home() / ".local/share/heymist/models"
    model_path = model_dir / f"ggml-{model_name}.bin"

    if not model_path.exists():
        log.info("Downloading whisper model '%s'...", model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        _download_whisper_model(model_name, model_path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        _write_wav(f, audio_bytes, config["audio"]["sample_rate"])

    try:
        result = subprocess.run(
            [
                whisper_bin,
                "--model", str(model_path),
                "--threads", str(config["whisper"]["threads"]),
                "--no-timestamps",
                "--language", "en",
                "--output-txt",
                "--output-file", tmp_path.replace(".wav", ""),
                tmp_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        txt_path = tmp_path.replace(".wav", ".txt")
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                text = f.read().strip()
            os.unlink(txt_path)
            return text
        else:
            return result.stdout.strip()
    except subprocess.TimeoutExpired:
        log.error("Whisper transcription timed out")
        return ""
    finally:
        os.unlink(tmp_path)


def _download_whisper_model(model_name, dest):
    """Download a whisper.cpp model from HuggingFace."""
    import urllib.request
    base_url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
    url = f"{base_url}/ggml-{model_name}.bin"
    log.info("Downloading %s ...", url)
    urllib.request.urlretrieve(url, dest)
    log.info("Model saved to %s", dest)


def transcribe_cohere_api(audio_bytes, config):
    """Transcribe using Cohere API."""
    import urllib.request

    api_key = config["cohere"].get("api_key") or os.environ.get(
        "COHERE_API_KEY", ""
    )
    if not api_key:
        log.error("No Cohere API key configured")
        return ""

    buf = io.BytesIO()
    _write_wav(buf, audio_bytes, config["audio"]["sample_rate"])
    wav_data = buf.getvalue()

    boundary = "----HeyMistBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + wav_data + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="model"\r\n\r\n'
        f"CohereLabs/cohere-transcribe-03-2026\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="language"\r\n\r\n'
        f"en\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    req = urllib.request.Request(
        "https://api.cohere.com/v2/audio/transcriptions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            return result.get("text", "")
    except Exception as e:
        log.error("Cohere API error: %s", e)
        return ""


BACKENDS = {
    "whisper-local": transcribe_whisper_local,
    "cohere-api": transcribe_cohere_api,
}


# ---------------------------------------------------------------------------
# Output methods
# ---------------------------------------------------------------------------

VOICE_COMMANDS = {
    "enter": [("key", "28:1", "28:0")],
    "press enter": [("key", "28:1", "28:0")],
    "submit": [("key", "28:1", "28:0")],
    "new line": [("key", "28:1", "28:0")],
    "newline": [("key", "28:1", "28:0")],
    "tab": [("key", "15:1", "15:0")],
    "escape": [("key", "1:1", "1:0")],
    "backspace": [("key", "14:1", "14:0")],
    "delete": [("key", "111:1", "111:0")],
    "undo": [("key", "29:1", "44:1", "44:0", "29:0")],
    "redo": [("key", "29:1", "42:1", "44:1", "44:0", "42:0", "29:0")],
    "select all": [("key", "29:1", "30:1", "30:0", "29:0")],
    "copy": [("key", "29:1", "46:1", "46:0", "29:0")],
    "paste": [("key", "29:1", "47:1", "47:0", "29:0")],
    "cut": [("key", "29:1", "45:1", "45:0", "29:0")],
    "save": [("key", "29:1", "31:1", "31:0", "29:0")],
}


def _run_ydotool(args):
    """Run ydotool with socket fallback via sg."""
    env = os.environ.copy()
    env["YDOTOOL_SOCKET"] = "/run/ydotoold/socket"
    result = subprocess.run(
        ["ydotool"] + args, env=env,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        import shlex
        log.warning("ydotool direct failed (%s), trying via sg", result.stderr.strip())
        cmd = "YDOTOOL_SOCKET=/run/ydotoold/socket ydotool " + " ".join(
            shlex.quote(a) for a in args
        )
        subprocess.run(["sg", "ydotool", "-c", cmd], check=True)


def output_ydotool(text):
    """Type text into active window via ydotool, with voice command support."""
    clean = text.lower().strip().rstrip(".,!?")

    # Strip prefix for voice command matching
    prefix = load_config().get("prefix", "")
    clean_no_prefix = clean
    if prefix and clean.startswith(prefix.lower()):
        clean_no_prefix = clean[len(prefix):].strip()

    if clean_no_prefix in VOICE_COMMANDS:
        for action in VOICE_COMMANDS[clean_no_prefix]:
            _run_ydotool(list(action))
        log.info("Executed voice command: %s", clean_no_prefix)
        return

    # Type the text and auto-submit
    _run_ydotool(["type", "--key-delay", "12", "--", text])
    time.sleep(0.05)
    _run_ydotool(["key", "28:1", "28:0"])
    log.info("Typed and submitted: %s", text)


def output_clipboard(text):
    """Copy to clipboard via wl-copy and notify."""
    subprocess.run(["wl-copy", "--", text], check=True)
    notify(f"Copied to clipboard: {text[:60]}...")


OUTPUT_METHODS = {
    "ydotool": output_ydotool,
    "clipboard": output_clipboard,
}


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


def notify(message):
    """Send a desktop notification."""
    try:
        subprocess.run(
            ["notify-send", "-a", "heymist", "-t", "3000", "heymist", message],
            check=False,
        )
    except FileNotFoundError:
        pass


def feedback_wake(config):
    """Signal that wake word was detected."""
    if not config["feedback"]["enabled"]:
        return
    notify("Listening...")


def feedback_transcribing(config):
    """Signal that we're transcribing."""
    if not config["feedback"]["enabled"]:
        return
    notify("Transcribing...")


def feedback_ready(config):
    """Signal that heymist is listening."""
    if not config["feedback"]["enabled"]:
        return
    notify("heymist is listening")


# ---------------------------------------------------------------------------
# Text-to-Speech
# ---------------------------------------------------------------------------

PIPER_MODEL = Path(os.environ.get(
    "HEYMIST_PIPER_MODEL",
    str(Path.home() / ".local/share/piper-voices/en_US-lessac-medium.onnx"),
))
SPEAK_FIFO = Path(os.environ.get(
    "HEYMIST_SPEAK_FIFO",
    str(Path.home() / ".local/share/heymist/speak.fifo"),
))

# Threading flag — pauses wake detection while TTS is playing
import threading
tts_active = threading.Event()


def speak(text):
    """Speak text aloud via piper TTS."""
    if not PIPER_MODEL.exists():
        log.warning("Piper model not found at %s, skipping TTS", PIPER_MODEL)
        return

    piper_bin = shutil.which("piper")
    if not piper_bin:
        log.warning("piper not found in PATH, skipping TTS")
        return

    # Truncate very long responses for sanity
    if len(text) > 2000:
        text = text[:2000] + "... truncated."

    log.info("Speaking: %s", text[:80])
    tts_active.set()  # pause wake word detection
    try:
        # piper → raw PCM → sox play
        piper_proc = subprocess.Popen(
            [piper_bin, "--model", str(PIPER_MODEL), "--output-raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        play_proc = subprocess.Popen(
            ["play", "-t", "raw", "-r", "22050", "-e", "signed", "-b", "16", "-c", "1", "-"],
            stdin=piper_proc.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        piper_proc.stdin.write(text.encode())
        piper_proc.stdin.close()
        play_proc.wait(timeout=60)
        piper_proc.wait(timeout=5)
    except Exception:
        log.exception("TTS error")
    finally:
        # Brief cooldown so mic doesn't pick up tail-end of speaker output
        time.sleep(0.5)
        tts_active.clear()
        log.debug("TTS finished, wake detection resumed")


def _tts_listener():
    """Background thread: read from FIFO and speak each line."""
    SPEAK_FIFO.parent.mkdir(parents=True, exist_ok=True)

    # Create FIFO if it doesn't exist
    if not SPEAK_FIFO.exists():
        os.mkfifo(SPEAK_FIFO)
        log.info("Created speak FIFO at %s", SPEAK_FIFO)

    log.info("TTS listener started on %s", SPEAK_FIFO)

    while True:
        try:
            # Open blocks until a writer connects
            with open(SPEAK_FIFO, "r") as f:
                text = f.read().strip()
                if text:
                    speak(text)
        except Exception:
            log.exception("TTS listener error")
            time.sleep(1)


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------


def _write_wav(file_or_path, audio_bytes, sample_rate):
    """Write raw int16 mono audio to WAV format."""
    with wave.open(file_or_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = load_config()

    backend_name = config["backend"]
    output_name = config["output"]
    prefix = config.get("prefix", "")

    transcribe_fn = BACKENDS.get(backend_name)
    if not transcribe_fn:
        log.error("Unknown backend: %s", backend_name)
        sys.exit(1)

    output_fn = OUTPUT_METHODS.get(output_name)
    if not output_fn:
        log.error("Unknown output method: %s", output_name)
        sys.exit(1)

    # Initialize wake word detector and command recorder
    detector = WakeWordDetector(config)
    recorder = CommandRecorder(config)

    # Start TTS listener in background thread
    import threading
    tts_thread = threading.Thread(target=_tts_listener, daemon=True)
    tts_thread.start()

    log.info(
        "heymist started (wakeword=%s, backend=%s, output=%s)",
        config["wakeword"]["model"],
        backend_name,
        output_name,
    )
    feedback_ready(config)

    while True:
        try:
            # Phase 1: Wait for wake word (lightweight, ~0 CPU)
            detector.listen_for_wake_word()
            feedback_wake(config)

            # Phase 2: Record command until silence
            audio_bytes = recorder.record_command()
            if not audio_bytes:
                continue

            # Phase 3: Transcribe
            feedback_transcribing(config)
            text = transcribe_fn(audio_bytes, config)
            if not text:
                continue

            # Filter whisper hallucinations
            text_clean = text.strip()
            if text_clean.startswith("[") or text_clean.startswith("("):
                log.debug("Filtered hallucination: %s", text_clean)
                continue

            log.info("Command: %s", text_clean)

            # Phase 4: Output
            output_fn(prefix + text_clean)

        except KeyboardInterrupt:
            log.info("Shutting down")
            break
        except Exception:
            log.exception("Error in main loop")
            time.sleep(1)


if __name__ == "__main__":
    main()
