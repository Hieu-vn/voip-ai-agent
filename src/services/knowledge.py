import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

class SimpleEmbeddingModel:
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
        self.vocabulary: Dict[str, int] = {}

    def _tokenize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        tokens = []
        token = []
        for ch in text:
            if ch.isalnum():
                token.append(ch)
            else:
                if token:
                    tokens.append(''.join(token))
                    token = []
        if token:
            tokens.append(''.join(token))
        return tokens

    def build_vocab(self, documents: List[str]) -> None:
        for doc in documents:
            for token in self._tokenize(doc):
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)

    def embed(self, text: str) -> List[float]:
        if not self.vocabulary:
            raise RuntimeError('Embedding vocabulary is empty; call build_vocab first.')
        counts = Counter(self._tokenize(text))
        vec = [0.0] * len(self.vocabulary)
        total = sum(counts.values()) or 1
        for token, count in counts.items():
            idx = self.vocabulary.get(token)
            if idx is not None:
                vec[idx] = count / total
        norm = math.sqrt(sum(v * v for v in vec))
        if norm:
            vec = [v / norm for v in vec]
        return vec

    @staticmethod
    def cosine(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))


class KnowledgeService:
    def __init__(self, base_path: Optional[Path] = None) -> None:
        kb_path = os.getenv('KNOWLEDGE_BASE_PATH')
        if kb_path:
            self.base_path = Path(kb_path)
        else:
            base = base_path or Path(__file__).resolve().parents[2]
            self.base_path = base / 'data' / 'knowledge_base.json'
        self.entries: List[Dict[str, Any]] = []
        self._embedding_model = SimpleEmbeddingModel()
        self._embeddings: List[List[float]] = []
        self._load_entries()

    def _load_entries(self) -> None:
        if not self.base_path.exists():
            raise FileNotFoundError(f'Knowledge base file not found: {self.base_path}')
        with self.base_path.open('r', encoding='utf-8') as fp:
            data = json.load(fp)
        if not isinstance(data, list):
            raise ValueError('Knowledge base must be a list of entries')
        self.entries = data
        questions = [entry.get('question', '') + ' ' + entry.get('answer', '') for entry in self.entries]
        self._embedding_model.build_vocab(questions)
        self._embeddings = [self._embedding_model.embed(text) for text in questions]

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.2) -> List[Dict[str, Any]]:
        if not query.strip() or not self.entries:
            return []
        query_vec = self._embedding_model.embed(query)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for entry, vec in zip(self.entries, self._embeddings):
            score = SimpleEmbeddingModel.cosine(query_vec, vec)
            scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for score, entry in scored[:top_k] if score >= min_score]

    def build_context(self, query: str, top_k: int = 3) -> str:
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return ''
        return '\n\n'.join(f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}" for item in results)
