import argparse
from pathlib import Path

from acoustic_topo_lock.core.config_loader import load_settings
from acoustic_topo_lock.core.exceptions import TopoLockError
from acoustic_topo_lock.core.logging_config import configure_logging
from acoustic_topo_lock.datasets.manifest import inspect_dataset, load_dataset


def train_command(kind: str):
    import json

    from acoustic_topo_lock.datasets.training import CANDIDATES, train

    parser = argparse.ArgumentParser(description=f"Inspect or train on a local {kind} manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--inspect", action="store_true", help="Validate file references and show counts.")
    parser.add_argument("--candidate", action="append", choices=CANDIDATES[kind])
    parser.add_argument("--parameters", type=Path, help="JSON object keyed by candidate name.")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    try:
        settings = load_settings()
        configure_logging(settings.log_level)
        dataset = load_dataset(args.manifest, kind)
        if args.inspect:
            print(json.dumps(inspect_dataset(dataset), indent=2))
            return
        parameters = json.loads(args.parameters.read_text()) if args.parameters else None
        metadata = train(
            dataset, settings, settings.model_root / kind,
            cache_dir=None if args.no_cache else settings.data_root / "features",
            candidates=args.candidate, parameters=parameters,
        )
        print(f"Exported {metadata['version']} to {settings.model_root / kind}")
    except (TopoLockError, OSError, ValueError, TypeError) as error:
        parser.exit(2, f"{error}\n")
    except ModuleNotFoundError as error:
        parser.exit(2, f"Install the research extra before fitting models ({error.name}).\n")
