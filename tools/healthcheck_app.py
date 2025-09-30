"""
Health check script for the main 'app' service.

This script is used by Docker's HEALTHCHECK instruction to determine if the
application container is healthy.

It performs two main checks:
1. Static Check: Verifies that essential libraries like CUDA and llama_cpp
   are installed and accessible.
2. Downstream Dependency Check: Verifies that it can connect to the TTS
   service, which is a critical dependency for the app to be functional.
"""
import http.client
import json
import os
import sys

def check_libraries():
    """Checks for the presence of critical libraries."""
    print("Checking for critical libraries...")
    try:
        import ctypes
        import llama_cpp
        # Check for CUDA driver
        ctypes.CDLL('libcuda.so.1')
        print(f"- CUDA driver found.")
        print(f"- llama_cpp version: {llama_cpp.__version__}")
        return True
    except Exception as e:
        print(f"Library check failed: {e}")
        return False

def check_tts_service():
    """Checks the health of the downstream TTS service."""
    print("Checking downstream TTS service health...")
    tts_host = os.getenv("TTS_SERVER_HOST", "localhost")
    tts_port = int(os.getenv("TTS_SERVER_PORT", 8001))
    
    try:
        conn = http.client.HTTPConnection(tts_host, tts_port, timeout=2)
        conn.request("GET", "/healthz")
        response = conn.getresponse()
        
        if response.status == 200:
            body = response.read().decode('utf-8')
            data = json.loads(body)
            if data.get("status") == "ok" and data.get("models_loaded") is True:
                print("- TTS service is healthy and models are loaded.")
                return True
        
        print(f"TTS service is unhealthy. Status: {response.status}, Body: {response.read().decode()}")
        return False
    except Exception as e:
        print(f"Failed to connect to TTS service: {e}")
        return False
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def main():
    """Run all health checks."""
    print("--- Running App Service Health Check ---")
    lib_ok = check_libraries()
    tts_ok = check_tts_service()

    if lib_ok and tts_ok:
        print("--- Health Check PASSED ---")
        sys.exit(0)
    else:
        print("--- Health Check FAILED ---")
        sys.exit(1)

if __name__ == "__main__":
    main()