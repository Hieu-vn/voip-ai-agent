"""
Evaluation tracking for logging conversation turns.

This module provides a simple tracker to log the details of each conversation
turn (user input, bot response, metadata) to a JSONL file. This data is
invaluable for later analysis, debugging, and model fine-tuning.
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import structlog

log = structlog.get_logger()

class EvaluationTracker:
    """Logs conversation turns to a file for later evaluation."""

    def __init__(self, log_dir: str = "data/eval_logs"):
        """
        Initializes the tracker and ensures the log directory exists.

        Args:
            log_dir: The directory where JSONL log files will be stored.
        """
        self.log_dir = Path(log_dir)
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log.info("Evaluation tracker initialized.", log_dir=str(self.log_dir))
        except Exception as e:
            log.error("Failed to create evaluation log directory.", path=str(self.log_dir), exc_info=e)

    def _get_log_file_path(self) -> Path:
        """Gets the path to today's log file."""
        today = datetime.utcnow().strftime('%Y-%m-%d')
        return self.log_dir / f'{today}.jsonl'

    def log_turn(
        self,
        session_id: str,
        turn_index: int,
        user_text: str,
        bot_text: str,
        metadata: Dict[str, Any],
    ):
        """
        Logs a single conversation turn to a file.

        Args:
            session_id: The unique ID for the call/session.
            turn_index: The index of this turn in the conversation (starting from 0).
            user_text: The text spoken by the user.
            bot_text: The text response from the bot.
            metadata: A dictionary of other useful data (intent, slots, latencies, etc.).
        """
        record = {
            'timestamp_utc': time.time(),
            'session_id': session_id,
            'turn_index': turn_index,
            'user_text': user_text,
            'bot_text': bot_text,
            'metadata': metadata,
        }
        
        try:
            log_file = self._get_log_file_path()
            with log_file.open('a', encoding='utf-8') as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as e:
            log.error("Failed to write to evaluation log.", record=record, exc_info=e)

# Create a singleton instance for easy import
evaluation_tracker = EvaluationTracker()
