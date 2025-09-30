from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class DialogState(BaseModel):
    call_id: str
    turn: int = 0
    last_user_text: str = ""
    last_bot_text: str = ""
    intent: Optional[str] = None
    slots: Dict[str, Any] = Field(default_factory=dict)
    emotion: Optional[str] = None
    awaiting_tool: Optional[str] = None
    done: bool = False