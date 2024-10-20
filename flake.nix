{
    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/nixos-23.11";
    };
    outputs = { nixpkgs, ... }: let
        inherit (nixpkgs) lib;
        eachSys = lib.genAttrs [ "x86_64-linux" ];
    in {
        devShells = eachSys (system: let
            pkgs = nixpkgs.legacyPackages.${system};
        in {
            default = pkgs.mkShell {
                LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver-32/lib";
                nativeBuildInputs = builtins.attrValues {
                    inherit (pkgs) pkg-config;
                };
                packages = builtins.attrValues {
                    inherit (pkgs) pyright gtk4 gtk2;
                    python3 = pkgs.python3.buildEnv.override {
                        extraLibs = builtins.attrValues {
                            inherit (pkgs.python3Packages)
                            glcontext
                            jinja2
                            moderngl
                            matplotlib
                            numpy
                            opencv4
                            pandas
                            pillow
                            pycairo
                            pyrr
                            sympy
                            ;
                        };
                    };
                };
            };
        });
    };
}
