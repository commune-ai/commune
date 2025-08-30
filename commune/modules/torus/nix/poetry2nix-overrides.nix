{ pkgs, p2n }:

let
  lib = pkgs.lib;

  pypkgs-build-requirements = {
    torus = [ "poetry-core" ];
    keylimiter = [ "poetry-core" ];
    scalecodec = [ "setuptools" ];
    streamlit = [ "setuptools" ];
    pyarrow-hotfix = [ "hatchling" ];
    gradio = [ "hatch-requirements-txt" "hatch-fancy-pypi-readme" ];
    gradio-client = [ "hatch-requirements-txt" "hatch-fancy-pypi-readme" ];
  };

  custom-overrides = self: super: {
    substrate-interface = super.substrate-interface.overridePythonAttrs (old: {
      # Set an environment variable for the version
      preBuild = ''
        export CI_COMMIT_TAG="v${old.version}"
      '';
      buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools ];
    });
    py-bip39-bindings =
      let
        getRepoHash = version: {
          "0.1.11" = "sha256-j1YPDMOQZU6xx1nkN5k7Wd/mqX0WWcrWL1p6AOHGZVE=";
        }.${version};
        sha256 = getRepoHash super.py-bip39-bindings.version;
      in
      super.py-bip39-bindings.overridePythonAttrs (old: lib.optionalAttrs (!(old.src.isWheel or false)) rec {
        src = pkgs.fetchFromGitHub {
          owner = "polkascan";
          repo = "py-bip39-bindings";
          # rev = "v${old.version}";
          rev = "310b3d93796493eedfb027c9a08eeb466e802f8b"; # hack to get version with Cargo.lock
          inherit sha256;
        };
        cargoDeps = pkgs.rustPlatform.importCargoLock {
          lockFile = "${src.out}/Cargo.lock";
        };
        buildInputs = (old.buildInputs or [ ]) ++ lib.optionals pkgs.stdenv.isDarwin [
          pkgs.libiconv
        ];
        nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
          pkgs.rustPlatform.cargoSetupHook
          pkgs.rustPlatform.maturinBuildHook
        ];
      });
    py-ed25519-zebra-bindings =
      let
        getRepoHash = version: {
          "1.0.1" = "sha256-Y3fTHrWH3J9v1vmanwR/eSMiJemchTUTfiyfNrCttsk=";
        }.${version};
        sha256 = getRepoHash super.py-ed25519-zebra-bindings.version;
      in
      super.py-ed25519-zebra-bindings.overridePythonAttrs (old: lib.optionalAttrs (!(old.src.isWheel or false)) rec {
        src = pkgs.fetchFromGitHub {
          owner = "polkascan";
          repo = "py-ed25519-zebra-bindings";
          # rev = "v${old.version}";
          rev = "4c8ad0740a093e3693053ed8ef7d4c1a4859c1dc"; # hack to get version with Cargo.lock
          inherit sha256;
        };
        cargoDeps = pkgs.rustPlatform.importCargoLock {
          lockFile = "${src.out}/Cargo.lock";
        };
        buildInputs = (old.buildInputs or [ ]) ++ lib.optionals pkgs.stdenv.isDarwin [
          pkgs.libiconv
        ];
        nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
          pkgs.rustPlatform.cargoSetupHook
          pkgs.rustPlatform.maturinBuildHook
        ];
      });
    py-sr25519-bindings =
      let
        getRepoHash = version: {
          "0.2.0" = "sha256-mxNmiFvMbV9WQhGNIQXxTkOcJHYs0vyOPM6Nd5367RE=";
        }.${version};
        sha256 = getRepoHash super.py-sr25519-bindings.version;
      in
      super.py-sr25519-bindings.overridePythonAttrs (old: lib.optionalAttrs (!(old.src.isWheel or false)) rec {
        src = pkgs.fetchFromGitHub {
          owner = "polkascan";
          repo = "py-sr25519-bindings";
          # rev = "v${old.version}";
          rev = "9127501235bf291d7f14f00ec373d0a5000a32cb"; # hack to get version with Cargo.lock
          inherit sha256;
        };
        cargoDeps = pkgs.rustPlatform.importCargoLock {
          lockFile = "${src.out}/Cargo.lock";
        };
        buildInputs = (old.buildInputs or [ ]) ++ lib.optionals pkgs.stdenv.isDarwin [
          pkgs.libiconv
        ];
        nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
          pkgs.rustPlatform.cargoSetupHook
          pkgs.rustPlatform.maturinBuildHook
        ];
      });
    safetensors =
      let
        getRepoHash = version: {
          "0.4.1" = "sha256-bxbJwfxctVvLMXIYbLusCyN0u7y4vhOUlYuKUgThdSo=";
        }.${version} or (
          lib.warn "Unknown safetensors version: '${version}'. Please update getCargoHash." lib.fakeHash
        );
      in
      super.safetensors.overridePythonAttrs (old: rec {
        src = (pkgs.fetchFromGitHub {
          owner = "steinerkelvin";
          repo = "safetensors";
          # rev = "v${old.version}";
          rev = "cd485b440c22bf2a0fa59315fa89584d6d903679"; # hack to get version with Cargo.lock
          sha256 = getRepoHash old.version;
        });
        sourceRoot = "${src.name}/bindings/python";
        nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [
          super.pdm-pep517
          pkgs.rustPlatform.cargoSetupHook
          pkgs.rustPlatform.maturinBuildHook
        ];
        cargoDeps = pkgs.rustPlatform.importCargoLock {
          lockFile = "${src.out}/bindings/python/Cargo.lock";
        };
      });
    sse-starlette = super.sse-starlette.overridePythonAttrs (old: {
      buildInputs = (old.buildInputs or [ ]) ++ [ super.pdm-pep517 super.pdm-backend ];
    });
  };

  simple-overrides = self: super: (builtins.mapAttrs
    (package: build-requirements:
      (builtins.getAttr package super).overridePythonAttrs (old: {
        buildInputs = (old.buildInputs or [ ]) ++ (builtins.map (pkg: if builtins.isString pkg then builtins.getAttr pkg super else pkg) build-requirements);
      })
    )
    pypkgs-build-requirements
  );

in

p2n.defaultPoetryOverrides.extend (
  self: super: (simple-overrides self super) // (custom-overrides self super)
)
