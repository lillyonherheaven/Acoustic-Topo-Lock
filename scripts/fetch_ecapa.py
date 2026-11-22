import argparse

from acoustic_topo_lock.core.config_loader import load_settings
from acoustic_topo_lock.core.model_registry import ECAPA_FILES, ECAPA_ID


def main():
    parser = argparse.ArgumentParser(description="Explicitly download official ECAPA weights for offline use.")
    parser.add_argument("--revision", default="0f99f2d0ebe89ac095bcc5903c4dd8f72b367286", help="Hugging Face model revision")
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        parser.exit(2, "Install requirements-ml.txt before fetching ECAPA.\n")
    directory = load_settings().model_root / "voice"
    snapshot_download(ECAPA_ID, revision=args.revision, local_dir=directory,
                      allow_patterns=list(ECAPA_FILES))
    missing = [name for name in ECAPA_FILES if not (directory / name).is_file()]
    if missing:
        parser.exit(2, f"Required files missing after download: {', '.join(missing)}\n")
    print(f"Installed {ECAPA_ID} revision {args.revision} under {directory}")


if __name__ == "__main__":
    main()
