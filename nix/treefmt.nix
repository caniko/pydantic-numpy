{pkgs, ...}: {
  projectRootFile = "flake.nix";

  programs.alejandra.enable = true;

  programs.taplo.enable = true;

  programs.prettier = {
    enable = true;
    package = pkgs.prettier;
    excludes = [
      ".crow/**"
    ];
    includes = [
      "*.md"
      "*.markdown"
      "*.yaml"
      "*.yml"
    ];
  };
}
