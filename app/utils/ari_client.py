import asyncio
import contextlib
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

import aiohttp
from aiohttp import ClientSession, WSMsgType
from urllib.parse import urlencode, urlparse, urlunparse

import structlog


log = structlog.get_logger()


class AriError(RuntimeError):
    """Raised when the ARI API returns an error."""


class AriClient:
    """Minimal asynchronous ARI client used by the VoIP agent."""

    def __init__(self, base_url: str, username: str, password: str, app_name: str) -> None:
        self._base_url = normalize_base_url(base_url)
        self._username = username
        self._password = password
        self._app_name = app_name

        self._session: Optional[ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._event_handlers: Dict[str, Callable[["Channel", Dict[str, Any]], Awaitable[None]]] = {}
        self.channels = ChannelsAPI(self)

    async def __aenter__(self) -> "AriClient":
        auth = aiohttp.BasicAuth(self._username, self._password)
        self._session = aiohttp.ClientSession(auth=auth)
        await self._open_websocket()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._ws_task:
            self._ws_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ws_task
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()

    # Public API ---------------------------------------------------------

    def on_channel_event(
        self,
        event_name: str,
        callback: Callable[["Channel", Dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Register a coroutine callback for a channel event (e.g. StasisStart)."""

        self._event_handlers[event_name] = callback

    # Internal helpers ---------------------------------------------------

    async def _open_websocket(self) -> None:
        if not self._session:
            raise RuntimeError("Client session is not initialised")

        ws_url = build_ws_url(self._base_url, self._app_name, self._username, self._password)
        log.info("Opening ARI WebSocket", url=ws_url)
        self._ws = await self._session.ws_connect(ws_url, heartbeat=10)
        self._ws_task = asyncio.create_task(self._event_loop())

    async def _event_loop(self) -> None:
        assert self._ws is not None
        async for message in self._ws:
            if message.type == WSMsgType.TEXT:
                await self._handle_event(message.data)
            elif message.type == WSMsgType.ERROR:
                log.warning("WebSocket error", error=self._ws.exception())
                break

    async def _handle_event(self, payload: str) -> None:
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            log.warning("Failed to decode ARI event", payload=payload)
            return

        event_type = event.get("type")
        if not event_type:
            return

        handler = self._event_handlers.get(event_type)
        if not handler:
            return

        channel_data = event.get("channel", {}) or {}
        channel = Channel(self, channel_data)

        try:
            result = handler(channel, event)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("Error handling ARI event", event=event_type, exc_info=exc)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if not self._session:
            raise RuntimeError("Client session is not initialised")

        url = f"{self._base_url}{path}"
        async with self._session.request(method.upper(), url, params=params, json=json_data) as response:
            if response.status >= 400:
                body = await response.text()
                raise AriError(f"ARI request failed: {response.status} {body}")

            if response.content_type == "application/json":
                return await response.json()
            return await response.text()


class ChannelsAPI:
    def __init__(self, client: AriClient) -> None:
        self._client = client

    async def externalMedia(self, **params: Any) -> "Channel":
        query: Dict[str, Any] = dict(params)
        # Align parameter names with ARI expectations
        if "channelId" in query:
            query["channel"] = query.pop("channelId")

        data = await self._client._request("post", "/ari/channels/externalMedia", params=query)
        if isinstance(data, str):
            data = json.loads(data)
        return Channel(self._client, data)


@dataclass
class Channel:
    client: AriClient
    data: Dict[str, Any]

    @property
    def id(self) -> Optional[str]:
        return self.data.get("id")

    @property
    def name(self) -> Optional[str]:
        return self.data.get("name")

    async def answer(self) -> None:
        if not self.id:
            raise AriError("Channel ID missing for answer() call")
        await self.client._request("post", f"/ari/channels/{self.id}/answer")

    async def hangup(self) -> None:
        if not self.id:
            return
        await self.client._request("delete", f"/ari/channels/{self.id}")


def build_ws_url(base_url: str, app_name: str, username: str, password: str) -> str:
    parsed = urlparse(base_url)
    path_prefix = parsed.path.rstrip('/')
    ws_path = f"{path_prefix}/ari/events"
    scheme = "ws" if parsed.scheme == "http" else "wss"
    query = urlencode(
        {
            "api_key": f"{username}:{password}",
            "app": app_name,
            "subscribeAll": "true",
        }
    )
    return urlunparse((scheme, parsed.netloc, ws_path, "", query, ""))


def normalize_base_url(url: str) -> str:
    stripped = url.rstrip('/')
    if stripped.lower().endswith('/ari'):
        stripped = stripped[: -len('/ari')]
    return stripped
