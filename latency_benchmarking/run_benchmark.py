"""Wrapper around benchmark_kitty.py that redirects quanto CUDA extension build
directory to a writable path before any quanto imports trigger compilation."""

import os
import sys

# Redirect quanto CUDA extension build dir to a writable location.
# The ext object is created at module import time but .lib is lazy —
# patch build_directory on the already-created object before first access.
_QUANTO_BUILD_DIR = "/data/jisenli2/quanto_cuda_build"

try:
    import optimum.quanto.library.extensions.cuda as _cuda_ext
    os.makedirs(_QUANTO_BUILD_DIR, exist_ok=True)
    _cuda_ext.ext.build_directory = os.path.join(_QUANTO_BUILD_DIR, "quanto_cuda")
except Exception:
    pass

# Run benchmark_kitty with original sys.argv
import runpy
sys.argv[0] = os.path.join(os.path.dirname(__file__), "benchmark_kitty.py")
runpy.run_path(sys.argv[0], run_name="__main__")
