from __future__ import annotations
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone
from pymongo.collection import Collection

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

class SessionRepo:
    def __init__(self, sessions: Collection):
        self.sessions = sessions

    def create_session(self, session_id: str, title: str = "New Session", session_config: Optional[Dict[str, Any]] = None) -> str:
        now = _utcnow()
        doc = {
            "_id": session_id,
            "created_at": now,
            "updated_at": now,
            "status": "active",
            "title": title,
            "session_config": session_config or {},
            "memory": {"summary": "", "constraints": {}, "last_updated_turn": 0},
            "counters": {"turn_count": 0},
            "pointers": {"last_turn_id": None},
        }
        self.sessions.insert_one(doc)
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.find_one({"_id": session_id})

    def touch(self, session_id: str, *, last_turn_id: Optional[str] = None, turn_count: Optional[int] = None) -> None:
        update: Dict[str, Any] = {"updated_at": _utcnow()}
        if last_turn_id is not None:
            update["pointers.last_turn_id"] = last_turn_id
        if turn_count is not None:
            update["counters.turn_count"] = turn_count
        self.sessions.update_one({"_id": session_id}, {"$set": update})

    def update_memory(self, session_id: str, summary: str, constraints: Dict[str, Any], last_updated_turn: int) -> None:
        self.sessions.update_one(
            {"_id": session_id},
            {"$set": {
                "updated_at": _utcnow(),
                "memory.summary": summary,
                "memory.constraints": constraints,
                "memory.last_updated_turn": last_updated_turn
            }}
        )

class TurnRepo:
    def __init__(self, turns: Collection):
        self.turns = turns

    def insert_turn(self, doc: Dict[str, Any]) -> str:
        self.turns.insert_one(doc)
        return doc["_id"]

    def get_last_turn(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.turns.find_one({"session_id": session_id}, sort=[("turn_index", -1)])

    def list_recent_turns(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        cur = self.turns.find({"session_id": session_id}).sort("turn_index", -1).limit(limit)
        return list(cur)[::-1]  # oldest→newest

    def get_turn(self, turn_id: str) -> Optional[Dict[str, Any]]:
        """Get a single turn by turn_id"""
        return self.turns.find_one({"_id": turn_id})

    def get_turns_by_session(self, session_id: str, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        """Get all turns for a session"""
        cur = self.turns.find(
            {"session_id": session_id}
        ).sort("created_at", -1).skip(skip).limit(limit)
        return list(cur)
