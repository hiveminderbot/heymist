{
  description = "heymist — voice interface for LLM coding agents";

  inputs = {
    nixpkgs.url = "https://flakehub.com/f/DeterminateSystems/nixpkgs-weekly/*";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = nixpkgs.legacyPackages.${system};
        inherit system;
      });
    in
    {
      packages = forAllSystems ({ pkgs, system }: let
        python = pkgs.python3.withPackages (ps: with ps; [
          sounddevice
          webrtcvad
          numpy
          pyyaml
          setuptools
          pyopen-wakeword
        ]);
      in {
        default = self.packages.${system}.heymist;

        heymist = pkgs.stdenv.mkDerivation {
          pname = "heymist";
          version = "0.1.0";
          src = ./.;

          nativeBuildInputs = [ pkgs.makeWrapper ];

          runtimeDeps = [
            python
            pkgs.whisper-cpp
            pkgs.piper-tts
            pkgs.ydotool
            pkgs.wl-clipboard
            pkgs.libnotify
            pkgs.sox
          ];

          installPhase = ''
            mkdir -p $out/lib/heymist $out/bin $out/share/heymist

            # Core Python source
            cp src/heymist.py $out/lib/heymist/
            cp src/calibrate.py $out/lib/heymist/

            # Default config
            cp default-config.yaml $out/share/heymist/

            # Main binary
            makeWrapper ${python}/bin/python3 $out/bin/heymist \
              --add-flags "$out/lib/heymist/heymist.py" \
              --prefix PATH : ${pkgs.lib.makeBinPath [
                pkgs.whisper-cpp
                pkgs.piper-tts
                pkgs.ydotool
                pkgs.wl-clipboard
                pkgs.libnotify
                pkgs.sox
              ]}

            # Calibration tool
            makeWrapper ${python}/bin/python3 $out/bin/heymist-calibrate \
              --add-flags "$out/lib/heymist/calibrate.py" \
              --prefix PATH : ${pkgs.lib.makeBinPath [ pkgs.whisper-cpp ]}

            # TTS utility
            cp bin/heymist-speak $out/bin/
            chmod +x $out/bin/heymist-speak

            # Claude Code hook
            cp bin/heymist-stop-hook $out/bin/
            chmod +x $out/bin/heymist-stop-hook
          '';

          meta = with pkgs.lib; {
            description = "Voice interface for LLM coding agents — wake word, STT, TTS, Wayland typing";
            license = licenses.mit;
            platforms = platforms.linux;
            mainProgram = "heymist";
          };
        };
      });

      # Home Manager module
      homeManagerModules.default = { config, lib, pkgs, ... }:
        let
          cfg = config.services.heymist;
          heymistPkg = self.packages.${pkgs.system}.heymist;
        in {
          options.services.heymist = {
            enable = lib.mkEnableOption "heymist voice interface daemon";

            package = lib.mkOption {
              type = lib.types.package;
              default = heymistPkg;
              description = "The heymist package to use.";
            };

            settings = lib.mkOption {
              type = lib.types.attrs;
              default = {};
              description = "heymist configuration (merged with defaults, written to config.yaml).";
              example = {
                wake_phrase = "hey mist";
                backend = "whisper-local";
                output = "ydotool";
                wakeword.model = "builtin:hey_jarvis";
                wakeword.threshold = 0.5;
                whisper.model = "small.en";
                prefix = "[voice] ";
              };
            };

            claudeCodeHook = lib.mkOption {
              type = lib.types.bool;
              default = false;
              description = "Install a Claude Code Stop hook for TTS responses to voice commands.";
            };
          };

          config = lib.mkIf cfg.enable {
            home.packages = [ cfg.package ];

            # Write merged config
            xdg.configFile."heymist/config.yaml".text =
              let
                defaults = builtins.fromJSON (builtins.readFile
                  (pkgs.runCommand "heymist-defaults" {} ''
                    ${pkgs.yq-go}/bin/yq -o json ${heymistPkg}/share/heymist/default-config.yaml > $out
                  ''));
                merged = lib.recursiveUpdate defaults cfg.settings;
              in
                builtins.toJSON merged;

            # Systemd user service
            systemd.user.services.heymist = {
              Unit = {
                Description = "heymist voice interface daemon";
                After = [ "graphical-session.target" "pipewire.service" ];
                Requires = [ "graphical-session.target" ];
              };

              Service = {
                Type = "simple";
                Environment = "YDOTOOL_SOCKET=/run/ydotoold/socket";
                ExecStart = "${cfg.package}/bin/heymist";
                Restart = "on-failure";
                RestartSec = 5;
              };

              Install = {
                WantedBy = [ "graphical-session.target" ];
              };
            };
          };
        };

      # Dev shell
      devShells = forAllSystems ({ pkgs, ... }: {
        default = pkgs.mkShell {
          packages = [
            (pkgs.python3.withPackages (ps: with ps; [
              sounddevice webrtcvad numpy pyyaml setuptools pyopen-wakeword
            ]))
            pkgs.whisper-cpp
            pkgs.piper-tts
            pkgs.ydotool
            pkgs.wl-clipboard
            pkgs.libnotify
            pkgs.sox
          ];
        };
      });
    };
}
