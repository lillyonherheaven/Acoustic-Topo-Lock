import argparse
import importlib.util
import json
import sys
from importlib.metadata import PackageNotFoundError, version

from acoustic_topo_lock.core.config_loader import load_settings
from acoustic_topo_lock.core.model_registry import ModelRegistry
from acoustic_topo_lock.core.status import Check

GROUPS = {
    "core": ("numpy", "soundfile", "joblib"),
    "ui": ("streamlit", "plotly"),
    "test": ("pytest",),
    "research": ("sklearn",),
    "voice": ("speechbrain", "torch", "torchaudio"),
}


def main():
    parser = argparse.ArgumentParser(description="Report runtime and local model availability.")
    parser.add_argument("--require", action="append", choices=GROUPS, default=["core"])
    args = parser.parse_args()
    settings = load_settings()
    registry = ModelRegistry(settings.model_root)
    packages = {}
    missing = []
    for group, names in GROUPS.items():
        packages[group] = {}
        for name in names:
            if importlib.util.find_spec(name) is None:
                packages[group][name] = None
                if group in args.require:
                    missing.append(name)
                continue
            try:
                packages[group][name] = version("scikit-learn" if name == "sklearn" else name)
            except PackageNotFoundError:
                packages[group][name] = "installed"
    supported = (3, 11) <= sys.version_info[:2] < (3, 14)
    print(json.dumps({
        "python": sys.version.split()[0], "supported_python": supported, "packages": packages,
        "models": {check.value: registry.inspect(check).status for check in Check},
        "missing_required": missing,
    }, indent=2))
    return 0 if supported and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
