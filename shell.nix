let
  nixpkgs = fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-26.05";
  pkgs = import nixpkgs { config = {}; overlays = []; };
in

pkgs.mkShell {
  packages = with pkgs; [
    cargo
    rustc
    rustfmt
    rust-analyzer
    cmake
    ninja
    ncurses
    valgrind
  ];
}
