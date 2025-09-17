import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

class EvaluationTracker:
    def __init__(self) -> None:
        default_dir = Path(__file__).resolve().parents[2] / 'data' / 'eval_logs'
        self.log_dir = Path(os.getenv('EVAL_LOG_DIR', str(default_dir)))
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _file_for_today(self) -> Path:
        today = datetime.utcnow().strftime('%Y-%m-%d')
        return self.log_dir / f'{today}.jsonl'

    def log_turn(self, session_id: str, turn_index: int, user_text: str, bot_text: str, metadata: Dict[str, Any]) -> None:
        record = {
            'ts': time.time(),
            'session_id': session_id,
            'turn_index': turn_index,
            'user_text': user_text,
            'bot_text': bot_text,
            'metadata': metadata,
        }
        log_file = self._file_for_today()
        with log_file.open('a', encoding='utf-8') as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + '\n')

    def summarize(self) -> Dict[str, Any]:
        log_file = self._file_for_today()
        stats = {
            'turns': 0,
            'avg_bot_length': 0.0,
            'avg_user_length': 0.0,
            'intent_counts': {},
        }
        if not log_file.exists():
            return stats
        total_bot = 0
        total_user = 0
        with log_file.open('r', encoding='utf-8') as fp:
            for line in fp:
                if not line.strip():
                    continue
                stats['turns'] += 1
                rec = json.loads(line)
                bot_text = rec.get('bot_text', '')
                user_text = rec.get('user_text', '')
                total_bot += len(bot_text)
                total_user += len(user_text)
                intent = rec.get('metadata', {}).get('intent', 'unknown')
                stats['intent_counts'][intent] = stats['intent_counts'].get(intent, 0) + 1
        if stats['turns']:
            stats['avg_bot_length'] = total_bot / stats['turns']
            stats['avg_user_length'] = total_user / stats['turns']
        return stats


evaluation_tracker = EvaluationTracker()
