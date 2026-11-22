import ast
import json
from pathlib import Path

from acoustic_topo_lock.application import AuthenticationService
from acoustic_topo_lock.core.contracts import DEMO_WARNING
from acoustic_topo_lock.core.status import Decision
from acoustic_topo_lock.utils.synthetic_audio import synthetic_wav


def main():
    root = Path(__file__).resolve().parents[1]
    required = ["app.py", "README.md", "requirements.txt", "requirements-ml.txt", "pyproject.toml",
                ".env.example", "data/README.md", "models/README.md", "results/README.md"]
    missing = [file for file in required if not (root / file).is_file()]
    if missing:
        raise SystemExit(f"Missing required files: {', '.join(missing)}")
    python_files = [root / "app.py"]
    for name in ("src", "interface", "training", "scripts", "tests"):
        python_files.extend((root / name).rglob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    notebooks = sorted((root / "notebooks").glob("[0-9][0-9]_*.ipynb"))
    if len(notebooks) != 7:
        raise SystemExit("Expected seven research notebooks.")
    for path in notebooks:
        content = json.loads(path.read_text(encoding="utf-8"))
        if content["nbformat"] != 4 or not any(c["cell_type"] == "code" for c in content["cells"]):
            raise SystemExit(f"Invalid notebook structure: {path.name}")
        for cell in content["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=str(path))
    result = AuthenticationService().authenticate_audio(synthetic_wav(), filename="synthetic.wav", mode="demo")
    if result.decision.decision is not Decision.REJECT or DEMO_WARNING not in result.warnings:
        raise SystemExit("Labeled demo smoke test failed.")
    print(f"Validated {len(python_files)} Python files, seven notebooks, configuration and demo routing.")


if __name__ == "__main__":
    main()
