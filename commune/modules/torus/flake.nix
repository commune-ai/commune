{
  description = "Torus official CLI / SDK";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs?ref=nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          system = system;
          config.allowUnfree = true;
        };
        p2n = poetry2nix.lib.mkPoetry2Nix {
          inherit pkgs;
        };
        p2n-overrides = import ./nix/poetry2nix-overrides.nix {
          inherit pkgs p2n;
        };
        python = pkgs.python310;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            python
            pkgs.poetry
            pkgs.ruff
          ];
        };
        packages = rec {
          torus = p2n.mkPoetryApplication {
            projectDir = ./.;
            python = python;
            overrides = p2n-overrides;
          };
          default = torus;
        };
      });
}
