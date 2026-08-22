"""Launcher: ``python -m sensenova_u1_rocm.launch <script.py> [args...]``

Applies the project's ROCm fixes, then runs the target script exactly as if
it had been invoked directly (same argv, same __main__).
"""
import runpy
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m sensenova_u1_rocm.launch <script.py> [args...]", file=sys.stderr)
        return 2
    import sensenova_u1_rocm

    target, *rest = sys.argv[1:]
    sys.argv = [target, *rest]
    if sensenova_u1_rocm._bypass_active:
        print("[sensenova_u1_rocm] MIOpen bypass active (conv via unfold+GEMM; "
              "SENU15_MIOPEN=1 to re-enable)", file=sys.stderr)
    runpy.run_path(target, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
