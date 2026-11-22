import json
import os
from pathlib import Path
from uuid import uuid4


def write_report(path: Path, report: dict) -> None:
    """Replace a JSON report atomically after it is fully serialized."""
    data = json.dumps(report, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(data, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
