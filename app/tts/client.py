"""
Asynchronous client for the TTS microservice.

This module provides a client class to interact with the TTS server. It is
designed to handle streaming audio responses, which is crucial for achieving
low-latency playback in the VoIP agent.
"""
import os
import aiohttp
import structlog
from typing import AsyncGenerator, Optional

log = structlog.get_logger()

class TTSClient:
    """A streaming client for the TTS synthesis API."""

    def __init__(self, base_url: Optional[str] = None):
        """
        Initializes the TTS client.

        Args:
            base_url: The base URL of the TTS server. Defaults to the value of
                      the TTS_SERVER_URL environment variable or http://localhost:8001.
        """
        self.base_url = base_url or os.getenv("TTS_SERVER_URL", "http://localhost:8001")
        self._session: Optional[aiohttp.ClientSession] = None
        log.info("TTSClient initialized", base_url=self.base_url)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Initializes and returns a thread-safe aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Calls the TTS server to synthesize text and streams the audio back.

        Args:
            text: The text to synthesize.

        Yields:
            Chunks of raw audio data (16-bit PCM @ 8kHz).
        """
        log.info("Requesting TTS synthesis stream", text=text)
        request_url = f"{self.base_url}/synthesize"
        payload = {"text": text}
        
        try:
            session = await self._get_session()
            async with session.post(request_url, json=payload) as response:
                response.raise_for_status()  # Raise exception for 4xx/5xx status codes
                
                # Stream the response content in chunks
                async for chunk in response.content.iter_any():
                    if chunk:
                        yield chunk
                log.info("TTS stream finished.")

        except aiohttp.ClientError as e:
            log.error("TTS synthesis failed due to a client error", exc_info=e)
            # In a production system, you might want to yield a pre-recorded
            # error message instead of raising an exception.
            raise
        except Exception as e:
            log.error("An unexpected error occurred during TTS synthesis", exc_info=e)
            raise

    async def close(self):
        """Closes the aiohttp client session."""
        if self._session and not self._session.closed:
            await self._session.close()
            log.info("TTSClient session closed.")
