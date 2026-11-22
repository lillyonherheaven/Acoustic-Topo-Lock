import argparse
from pathlib import Path

from acoustic_topo_lock.core.config_loader import load_settings
from acoustic_topo_lock.core.exceptions import TopoLockError
from acoustic_topo_lock.core.model_registry import ModelRegistry
from acoustic_topo_lock.core.status import Check
from acoustic_topo_lock.datasets.manifest import load_dataset
from acoustic_topo_lock.datasets.training import evaluate
from acoustic_topo_lock.evaluation.reports import write_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate a frozen model once on the declared eval split.")
    parser.add_argument("kind", choices=("room", "replay"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    try:
        settings = load_settings()
        dataset = load_dataset(args.manifest, args.kind)
        record = ModelRegistry(settings.model_root).inspect(Check(args.kind))
        report = evaluate(dataset, settings, record,
                          cache_dir=None if args.no_cache else settings.data_root / "features")
        output = args.output or Path("results/reports") / f"{args.kind}-eval.json"
        write_report(output, report)
        print(f"Wrote {report['trial_count']} evaluation trials to {output}")
    except (TopoLockError, OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
