import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def case_root(case_id: str) -> Path:
    return ROOT / "data" / case_id


def v5_root(case_id: str) -> Path:
    return case_root(case_id) / "v5"


def session_root(case_id: str, session_id: str) -> Path:
    return v5_root(case_id) / "sessions" / session_id


def ensure_session_dirs(case_id: str, session_id: str) -> Path:
    base = session_root(case_id, session_id)
    for rel in [
        "raw",
        "logs",
        "asr",
        "alignment",
        "reflection",
        "error-slices",
        "detector-plans/errors",
        "review",
    ]:
        (base / rel).mkdir(parents=True, exist_ok=True)
    return base


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_json_list(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def resolve_path(path_value: str | Path) -> Path:
    p = Path(path_value)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path.resolve())


def extract_json_from_text(text: str) -> Any:
    fenced = re.search(r"```(?:json)?\s*([\[{][\s\S]*[\]}])\s*```", text)
    if fenced:
        return json.loads(fenced.group(1))
    starts = [idx for idx in [text.find("{"), text.find("[")] if idx >= 0]
    if not starts:
        raise ValueError("no json found in text")
    start = min(starts)
    end_obj = text.rfind("}")
    end_arr = text.rfind("]")
    end = max(end_obj, end_arr)
    if end <= start:
        raise ValueError("no complete json found in text")
    return json.loads(text[start : end + 1])


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def dedupe_str(values: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    if not isinstance(values, list):
        return out
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out
