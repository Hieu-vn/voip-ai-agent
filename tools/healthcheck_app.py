import ctypes, os
import llama_cpp
import sys

try:
    # Check CUDA driver availability
    ctypes.CDLL('libcuda.so.1')
    # Check llama_cpp import
    _ = llama_cpp.__version__
    print(f"App Healthcheck successful. CUDA driver loaded. llama_cpp version: {llama_cpp.__version__}")
    sys.exit(0)
except Exception as e:
    print(f"App Healthcheck failed: {e}")
    sys.exit(1)
