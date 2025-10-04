import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if "qdrant_client" not in sys.modules:
    class _StubQdrantClient:  # pragma: no cover - test-only scaffold
        def __init__(self, *_, **__):
            pass

        def search(self, *_, **__):  # noqa: D401
            """Return empty results to satisfy optional dependency."""
            return []

    sys.modules["qdrant_client"] = types.SimpleNamespace(QdrantClient=_StubQdrantClient)
