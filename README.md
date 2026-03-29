# heymist

Voice interface for LLM coding agents on Linux/Wayland.

Say a wake word, speak your command, and it gets typed into your active window with an automatic Enter. Responses from LLM agents are spoken back to you via text-to-speech.

## Features

- **Wake word detection** via [openwakeword](https://github.com/dscripka/openwakeword) — lightweight neural net, near-zero CPU when idle
- **Speech-to-text** via [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — runs locally on CPU, no API keys needed
- **Text-to-speech** via [piper](https://github.com/rhasspy/piper) — fast neural TTS, streams to PipeWire/PulseAudio
- **Wayland-native typing** via [ydotool](https://github.com/ReimuNotMoe/ydotool) — types into any focused window
- **LLM integration** — Claude Code Stop hook speaks responses to voice commands
- **Configurable** — swap STT/TTS backends, wake words, output methods, and more

## Requirements

- Linux with PipeWire or PulseAudio
- Wayland (GNOME, KDE, Sway, etc.) with ydotool for typing
- A microphone
- ~500MB disk for whisper small.en + piper voice model

No GPU required. Runs entirely on CPU.

## Quick Start (NixOS)

### As a flake input

```nix
# flake.nix
{
  inputs.heymist.url = "github:your-user/heymist";

  outputs = { heymist, ... }: {
    homeConfigurations."you" = home-manager.lib.homeManagerConfiguration {
      modules = [
        heymist.homeManagerModules.default
        {
          services.heymist = {
            enable = true;
            settings = {
              wakeword.model = "builtin:hey_jarvis";  # or path to custom .tflite
              whisper.model = "small.en";
              output = "ydotool";
              prefix = "[voice] ";
            };
          };
        }
      ];
    };
  };
}
```

### Try it without installing

```bash
nix run github:your-user/heymist
```

### System requirements for ydotool

ydotool needs a system-level daemon and uinput access. On NixOS:

```nix
# configuration.nix
programs.ydotool.enable = true;
users.users.your-user.extraGroups = [ "ydotool" ];
```

## Configuration

Config lives at `~/.config/heymist/config.yaml`:

```yaml
# Wake word detection (openwakeword)
wakeword:
  model: "builtin:hey_jarvis"  # or path to custom .tflite
  threshold: 0.5

# Speech-to-text
backend: whisper-local
whisper:
  model: small.en
  threads: 4

# Output method
output: ydotool  # or "clipboard"
prefix: "[voice] "

# Audio settings
audio:
  sample_rate: 16000
  vad_threshold: 0.015
  silence_duration: 2.0
  min_speech: 0.5
  max_duration: 30

# Text-to-speech
feedback:
  enabled: true
  type: notify
```

## Voice Models

### Whisper (STT)

Models are auto-downloaded on first use to `~/.local/share/heymist/models/`.

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `base.en` | 142MB | Fast | Good for commands |
| `small.en` | 466MB | Medium | Better accuracy |
| `medium.en` | 1.5GB | Slower | Best accuracy |

### Piper (TTS)

Download a voice model:

```bash
mkdir -p ~/.local/share/piper-voices
cd ~/.local/share/piper-voices
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

### Custom Wake Word

Train a custom wake word model using the [openwakeword Colab notebook](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb). It generates synthetic training data automatically — no recordings needed. Place the resulting `.tflite` file in `~/.local/share/heymist/models/` and update config:

```yaml
wakeword:
  model: "~/.local/share/heymist/models/hey_mist.tflite"
```

## Claude Code Integration

heymist includes a Stop hook that speaks Claude's responses when your input was a voice command (detected by the `[voice]` prefix).

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [{
      "matcher": "",
      "hooks": [{
        "type": "command",
        "command": "heymist-stop-hook"
      }]
    }]
  }
}
```

## Utilities

```bash
# Speak any text
echo "Hello world" | heymist-speak
heymist-speak "Hello world"

# Calibrate VAD thresholds for your mic/environment
heymist-calibrate

# Check service status
systemctl --user status heymist
journalctl --user -u heymist -f
```

## Architecture

```
Mic → openwakeword (wake detection) → sounddevice (record command)
    → whisper.cpp (transcribe) → ydotool (type into window)

FIFO ← heymist-speak / hooks → piper (TTS) → sox play → speakers
```

## License

MIT
