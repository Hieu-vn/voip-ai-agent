"""
VoIP AI Agent - Main Application Entrypoint (ARI)

This module is the core of the 'app' service. It connects to the Asterisk
REST Interface (ARI) via a WebSocket, listens for incoming calls, and hands
them off to a dedicated handler for processing.

Architecture:
- Uses `asyncio` for concurrent handling of multiple calls.
- Uses `ari-py` library to interact with the Asterisk ARI.
- Implements a persistent WebSocket client to receive real-time call events.
- For each incoming call (`StasisStart` event), it spawns a `CallHandler` task.
- Integrates `structlog` for structured, context-aware logging.
- Sets up OpenTelemetry for distributed tracing.
"""
import asyncio
import logging
import os
import signal

import structlog
from opentelemetry import trace
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor

# Import the new call handler (to be created)
from app.audio.stream import CallHandler
from app.utils.ari_client import AriClient

# --- Configuration ---
# Configure logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()

# Configure tracing
AioHttpClientInstrumentor().instrument()
tracer = trace.get_tracer(__name__)

# --- ARI Connection Details ---
ARI_URL = os.getenv("ARI_URL", "http://localhost:8088/")
ARI_USERNAME = os.getenv("ARI_USERNAME", "ai_user")
ARI_PASSWORD = os.getenv("ARI_PASSWORD", "supersecret")
ARI_APP_NAME = os.getenv("ARI_APP_NAME", "voip-ai-agent") # Must match the Stasis app name in extensions.conf

# A placeholder for the call handler class that will be implemented in stream.py
async def on_stasis_start(channel, event):
    """
    Callback for when a new channel enters the Stasis (our ARI) application.
    This is the entry point for a new call.
    """
    call_id = event['channel']['id']
    structlog.contextvars.bind_contextvars(call_id=call_id)
    log.info("StasisStart event received for new call", channel_name=event['channel']['name'])
    
    # Create and run a handler for this specific call
    handler = CallHandler(channel.client, channel)
    asyncio.create_task(handler.handle_call())

async def main():
    """
    Main asynchronous function to connect to ARI and run the application.
    """
    shutdown_event = asyncio.Event()

    def signal_handler():
        log.info("Shutdown signal received.")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    while not shutdown_event.is_set():
        try:
            log.info("Connecting to ARI...", url=ARI_URL, app=ARI_APP_NAME)
            async with AriClient(ARI_URL, ARI_USERNAME, ARI_PASSWORD, ARI_APP_NAME) as client:
                log.info("ARI connection successful. Listening for calls.")
                
                # Subscribe to the StasisStart event for our app
                client.on_channel_event("StasisStart", on_stasis_start)
                
                # Keep the connection alive until a shutdown signal is received
                await shutdown_event.wait()
                log.info("Disconnecting from ARI.")
                
        except asyncio.CancelledError:
            log.info("Main task cancelled.")
            break
        except Exception as e:
            log.error("ARI connection failed. Retrying in 5 seconds...", exc_info=e)
            if not shutdown_event.is_set():
                await asyncio.sleep(5)

if __name__ == "__main__":
    log.info("Starting VoIP AI Agent - App Service")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Application stopped by user.")
