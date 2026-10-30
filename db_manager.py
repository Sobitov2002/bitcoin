"""
db_manager.py — MongoDB and Local Fallback Manager for Bitcoin AI Dashboard.

Manages saving, retrieving, and evaluating AI predictions and user market observations.
Safely falls back to local storage if MongoDB is not yet configured or temporarily offline.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent

# Load environment variables safely
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
        except Exception:
            pass

MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "bitcoin_ai_db")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "observations")
LOCAL_BACKUP_PATH = BASE_DIR / "data" / "observations_backup.json"

# In-memory cached mongo client
_mongo_client = None
_db = None
_collection = None


def _is_placeholder_uri(uri: str) -> bool:
    """Check if the URI still contains placeholder tokens like <db_username>."""
    if not uri:
        return True
    return "<" in uri or ">" in uri or "db_username" in uri


def get_mongo_collection():
    """Lazily connect to MongoDB with a short timeout. Return None if unavailable."""
    global _mongo_client, _db, _collection

    if _is_placeholder_uri(MONGO_URI):
        return None

    if _collection is not None:
        return _collection

    try:
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
            socketTimeoutMS=3000,
        )
        # Test connection
        client.admin.command("ping")

        db = client[MONGO_DB_NAME]
        col = db[MONGO_COLLECTION]

        # Ensure indexes
        try:
            col.create_index("status")
            col.create_index("created_at_utc")
            col.create_index("target_time_utc")
        except Exception:
            pass

        _mongo_client = client
        _db = db
        _collection = col
        return _collection
    except Exception:
        return None


def check_connection() -> Dict[str, Any]:
    """Test connection status to MongoDB and return detailed diagnostic info."""
    if not MONGO_URI:
        return {
            "connected": False,
            "status": "NOT_CONFIGURED",
            "message": ".env faylida MONGO_URI topilmadi.",
            "mode": "LOCAL_STORAGE",
        }

    if _is_placeholder_uri(MONGO_URI):
        return {
            "connected": False,
            "status": "PLACEHOLDER_CREDENTIALS",
            "message": ".env faylida <db_username> hali haqiqiy foydalanuvchi nomi bilan almashtirilmagan.",
            "mode": "LOCAL_STORAGE",
        }

    try:
        col = get_mongo_collection()
        if col is not None:
            doc_count = col.count_documents({})
            return {
                "connected": True,
                "status": "CONNECTED",
                "message": f"MongoDB ulandi. Jami kuzatuvlar: {doc_count}",
                "mode": "MONGODB",
                "count": doc_count,
            }
    except Exception as exc:
        return {
            "connected": False,
            "status": "CONNECTION_FAILED",
            "message": f"MongoDB ga ulanib bo'lmadi: {exc}",
            "mode": "LOCAL_STORAGE",
        }

    return {
        "connected": False,
        "status": "UNKNOWN_ERROR",
        "message": "MongoDB ulanishini o'rnatib bo'lmadi.",
        "mode": "LOCAL_STORAGE",
    }


# =========================================================
# LOCAL STORAGE FALLBACK HELPERS
# =========================================================
def _load_local_records() -> List[Dict[str, Any]]:
    if not LOCAL_BACKUP_PATH.exists():
        return []
    try:
        with open(LOCAL_BACKUP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_local_records(records: List[Dict[str, Any]]) -> None:
    LOCAL_BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCAL_BACKUP_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)


# =========================================================
# OBSERVATION OPERATIONS
# =========================================================
def save_observation(doc: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """Save an observation document.

    First attempts MongoDB; if unavailable, stores in local JSON backup.
    """
    # Ensure ID and timestamps
    if "_id" not in doc or not doc["_id"]:
        doc["_id"] = str(uuid.uuid4())

    now_utc = datetime.now(timezone.utc)
    if "created_at_utc" not in doc:
        doc["created_at_utc"] = now_utc.isoformat()

    doc["status"] = doc.get("status", "PENDING")

    col = get_mongo_collection()
    if col is not None:
        try:
            col.insert_one(doc.copy())
            # Also keep local backup copy for safety
            records = _load_local_records()
            records.append(doc)
            _save_local_records(records)
            return True, "Kuzatuv MongoDB va lokal xotiraga muvaffaqiyatli saqlandi.", doc
        except Exception as exc:
            # Fall back to local
            records = _load_local_records()
            records.append(doc)
            _save_local_records(records)
            return True, f"MongoDB xatosi tufayli kuzatuv lokal saqlandi: {exc}", doc

    # Offline / local storage fallback
    records = _load_local_records()
    records.append(doc)
    _save_local_records(records)
    return True, "Kuzatuv lokal bazaga saqlandi (MongoDB ulanmagan).", doc


def get_all_observations(status: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    """Fetch observations sorted by created_at_utc descending."""
    col = get_mongo_collection()
    if col is not None:
        try:
            query = {"status": status} if status else {}
            cursor = col.find(query).sort("created_at_utc", -1).limit(limit)
            return list(cursor)
        except Exception:
            pass

    # Read from local
    records = _load_local_records()
    if status:
        records = [r for r in records if r.get("status") == status]
    records.sort(key=lambda r: str(r.get("created_at_utc", "")), reverse=True)
    return records[:limit]


def get_pending_observations() -> List[Dict[str, Any]]:
    """Fetch all observations whose status is PENDING."""
    return get_all_observations(status="PENDING", limit=500)


def update_observation_evaluation(obs_id: str, evaluation: Dict[str, Any], new_status: str = "COMPLETED") -> bool:
    """Update the evaluation results of an observation."""
    col = get_mongo_collection()
    updated = False

    if col is not None:
        try:
            res = col.update_one(
                {"_id": obs_id},
                {"$set": {"evaluation": evaluation, "status": new_status}}
            )
            updated = (res.modified_count > 0 or res.matched_count > 0)
        except Exception:
            updated = False

    # Also update local records
    records = _load_local_records()
    for r in records:
        if str(r.get("_id")) == str(obs_id):
            r["evaluation"] = evaluation
            r["status"] = new_status
            updated = True
            break
    _save_local_records(records)

    return updated


def calculate_accuracy_stats() -> Dict[str, Any]:
    """Calculate thorough, fair comparison stats between AI forecast and User predictions.

    Excludes pending or error records. Separates 1-hour and 4-hour performance.
    """
    observations = get_all_observations(limit=1000)

    total_count = len(observations)
    pending_count = sum(1 for o in observations if o.get("status") == "PENDING")
    error_count = sum(1 for o in observations if o.get("status") == "ERROR")
    completed = [o for o in observations if o.get("status") == "COMPLETED" and o.get("evaluation")]

    completed_count = len(completed)

    if completed_count == 0:
        return {
            "total_observations": total_count,
            "pending_count": pending_count,
            "error_count": error_count,
            "completed_count": 0,
            "ai_accuracy": None,
            "user_accuracy": None,
            "ai_1h_accuracy": None,
            "user_1h_accuracy": None,
            "ai_4h_accuracy": None,
            "user_4h_accuracy": None,
            "avg_abs_price_error": None,
            "avg_abs_pct_error": None,
        }

    ai_correct_total = 0
    user_correct_total = 0

    h1_completed = [o for o in completed if int(o.get("horizon_hours", 4)) == 1]
    h4_completed = [o for o in completed if int(o.get("horizon_hours", 4)) == 4]

    price_errors = []
    pct_errors = []

    for o in completed:
        ev = o.get("evaluation", {})
        if ev.get("ai_correct") is True:
            ai_correct_total += 1
        if ev.get("user_correct") is True:
            user_correct_total += 1

        price_diff = ev.get("price_change")
        if price_diff is not None:
            price_errors.append(abs(float(price_diff)))
        pct_diff = ev.get("price_change_pct")
        if pct_diff is not None:
            pct_errors.append(abs(float(pct_diff)))

    def _calc_acc(subset, key):
        valid = [o for o in subset if o.get("evaluation", {}).get(key) is not None]
        if not valid:
            return None
        correct = sum(1 for o in valid if o["evaluation"][key] is True)
        return (correct / len(valid)) * 100.0

    return {
        "total_observations": total_count,
        "pending_count": pending_count,
        "error_count": error_count,
        "completed_count": completed_count,
        "ai_accuracy": (ai_correct_total / completed_count) * 100.0,
        "user_accuracy": (user_correct_total / completed_count) * 100.0,
        "ai_1h_accuracy": _calc_acc(h1_completed, "ai_correct"),
        "user_1h_accuracy": _calc_acc(h1_completed, "user_correct"),
        "ai_4h_accuracy": _calc_acc(h4_completed, "ai_correct"),
        "user_4h_accuracy": _calc_acc(h4_completed, "user_correct"),
        "avg_abs_price_error": (sum(price_errors) / len(price_errors)) if price_errors else None,
        "avg_abs_pct_error": (sum(pct_errors) / len(pct_errors)) if pct_errors else None,
        "h1_count": len(h1_completed),
        "h4_count": len(h4_completed),
    }
