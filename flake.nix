{
  description = "pydantic-numpy: Python uv project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    py-harbor = {
      url = "git+https://github.com/caniko/harbor-py.git?ref=trunk&rev=a9396b7334869576ab46191ca7b34a9c63878be5";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    treefmt-nix.url = "github:numtide/treefmt-nix";
    git-hooks.url = "github:cachix/git-hooks.nix";
  };

  outputs = {
    self,
    py-harbor,
    treefmt-nix,
    git-hooks,
    ...
  }: let
    py = py-harbor.lib;

    mkDevShells = system: let
      pkgs = py.mkPkgs {inherit system;};
      treefmtEval = treefmt-nix.lib.evalModule pkgs (import ./nix/treefmt.nix);
      pre-commit-check = git-hooks.lib.${system}.run {
        src = ./.;
        hooks = import ./nix/pre-commit.nix {
          inherit pkgs;
          treefmtWrapper = treefmtEval.config.build.wrapper;
        };
      };
    in {
      default = pkgs.mkShell {
        packages = [pkgs.python314 pkgs.uv pkgs.just] ++ pre-commit-check.enabledPackages;
        env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc.lib];
        shellHook = ''
          unset PYTHONPATH
          uv sync --locked --group dev
          . .venv/bin/activate
          ${pre-commit-check.shellHook}
        '';
      };
    };

    mkPythonPackage = system: let
      pkgs = py.mkPkgs {inherit system;};
    in
      py.mkUvAppPackage {
        inherit pkgs;
        python = pkgs.python314;
        name = "pydantic-numpy";
        workspaceRoot = ./.;
        dependencies = {
          "pydantic-numpy" = [];
        };
        scripts = ["python"];
      };

    mkPythonCheckEnv = system: let
      pkgs = py.mkPkgs {inherit system;};
    in
      py.mkUvCheckEnv {
        inherit pkgs;
        python = pkgs.python314;
        name = "pydantic-numpy-check";
        workspaceRoot = ./.;
        dependencies = {
          "pydantic-numpy" = ["dev"];
        };
      };

    mkChecks = system: let
      pkgs = py.mkPkgs {inherit system;};
      checkEnv = mkPythonCheckEnv system;
      package = self.packages.${system}.default;
    in {
      flake-eval = pkgs.runCommand "pydantic-numpy-flake-eval" {} ''
        test -x ${package}/bin/python
        mkdir -p $out
        echo ok > $out/result
      '';
      formatting = let
        treefmtEval = treefmt-nix.lib.evalModule pkgs (import ./nix/treefmt.nix);
      in
        treefmtEval.config.build.check self;
      offline-tests = pkgs.runCommand "pydantic-numpy-offline-tests" {} ''
        export HOME=$TMPDIR/home
        export XDG_CACHE_HOME=$TMPDIR/cache
        mkdir -p "$HOME" "$XDG_CACHE_HOME" "$out"
        cd ${./.}
        ${checkEnv}/bin/python -m pytest tests -p no:cacheprovider
        echo ok > $out/result
      '';
      typecheck = pkgs.runCommand "pydantic-numpy-typecheck" {} ''
        export HOME=$TMPDIR/home
        export XDG_CACHE_HOME=$TMPDIR/cache
        mkdir -p "$HOME" "$XDG_CACHE_HOME" "$out"
        cd ${./.}
        ${checkEnv}/bin/python -m mypy .
        ${checkEnv}/bin/pyright --pythonpath ${checkEnv}/bin/python .
        echo ok > $out/result
      '';
      uv-format = pkgs.runCommand "pydantic-numpy-uv-format" {} ''
        export HOME=$TMPDIR/home
        export XDG_CACHE_HOME=$TMPDIR/cache
        export RUFF_CACHE_DIR=$TMPDIR/ruff-cache
        mkdir -p "$HOME" "$XDG_CACHE_HOME" "$out"
        cd ${./.}
        ${checkEnv}/bin/ruff format --check .
        echo ok > $out/result
      '';
    };
  in {
    devShells = py.forAllSystems mkDevShells;

    packages = py.forPackageSystems (
      system: {
        default = mkPythonPackage system;
      }
    );

    formatter = py.forAllSystems (
      system: let
        pkgs = py.mkPkgs {inherit system;};
        treefmtEval = treefmt-nix.lib.evalModule pkgs (import ./nix/treefmt.nix);
      in
        treefmtEval.config.build.wrapper
    );

    checks = py.forPackageSystems mkChecks;
  };
}
