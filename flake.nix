{
  description = "pydantic-numpy: Python uv project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    harbor = {
      url = "git+https://github.com/caniko/harbor.git?ref=feat/harbor-monorepo-components&rev=0a7e47bf3c6b6dfb45db18cb5b1b8bc8bd94523d";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    treefmt-nix.url = "github:numtide/treefmt-nix";
    git-hooks.url = "github:cachix/git-hooks.nix";
    # git-hooks imports its own package set, including on Intel macOS.
    git-hooks.inputs.nixpkgs.follows = "harbor/nixpkgs-darwin";
  };

  outputs = {
    self,
    harbor,
    treefmt-nix,
    git-hooks,
    ...
  }: let
    py = harbor.lib.python;
    mkTreefmt = pkgs:
      treefmt-nix.lib.evalModule pkgs {
        imports = [harbor.treefmtModules.python-python ./nix/treefmt.nix];
      };

    mkDevShells = system: let
      pkgs = py.mkPkgs {inherit system;};
      treefmtEval = mkTreefmt pkgs;
      pre-commit-check = git-hooks.lib.${system}.run {
        src = ./.;
        hooks = import ./nix/pre-commit.nix {
          inherit pkgs;
          treefmtWrapper = treefmtEval.config.build.wrapper;
          harborHooks = harbor.lib.core.hooks;
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
        treefmtEval = mkTreefmt pkgs;
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
      # Preserve the public gate while using the shared uncached formatter.
      uv-format = self.checks.${system}.formatting;
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
        treefmtEval = mkTreefmt pkgs;
      in
        treefmtEval.config.build.wrapper
    );

    checks = py.forPackageSystems mkChecks;
  };
}
