"""
Health check script for the TTS service.

This script is intended to be used by Docker's HEALTHCHECK instruction.
It sends a request to the /healthz endpoint of the running TTS server
and exits with a status code of 0 for success or 1 for failure.
"""
import http.client
import json
import sys

# Configuration
HOST = "localhost"
PORT = 8001  # Must match the port the TTS server runs on
TIMEOUT = 2  # seconds

def main():
    """Performs the health check.
    """
    try:
        # Use http.client to avoid external dependencies
        conn = http.client.HTTPConnection(HOST, PORT, timeout=TIMEOUT)
        conn.request("GET", "/healthz")
        response = conn.getresponse()

        # 1. Check for a successful HTTP status code
        if response.status != 200:
            print(f"Health check failed: Received HTTP status {response.status}")
            sys.exit(1)

        # 2. Check the content of the response
        body = response.read().decode('utf-8')
        data = json.loads(body)

        if data.get("status") == "ok" and data.get("models_loaded") is True:
            print("Health check passed: Server is ready and models are loaded.")
            sys.exit(0)
        else:
            print(f"Health check failed: Server is running but not ready. Response: {data}")
            sys.exit(1)

    except ConnectionRefusedError:
        print("Health check failed: Connection refused. Server is not running.")
        sys.exit(1)
    except http.client.HTTPException as e:
        print(f"Health check failed: HTTP error occurred. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during health check: {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    main()
