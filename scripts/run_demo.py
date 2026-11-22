import argparse
import json

from acoustic_topo_lock.application import AuthenticationService
from acoustic_topo_lock.application.demo import DEMO_CASES
from acoustic_topo_lock.application.result_formatter import result_dict
from acoustic_topo_lock.utils.synthetic_audio import synthetic_wav


def main():
    parser = argparse.ArgumentParser(description="Run the labeled synthetic software demonstration.")
    parser.add_argument("--case", choices=DEMO_CASES, default="Room matches, replay detected")
    args = parser.parse_args()
    result = AuthenticationService().authenticate_audio(
        synthetic_wav(), mode="demo", filename="synthetic.wav", demo_case=args.case,
    )
    print(json.dumps(result_dict(result), indent=2))


if __name__ == "__main__":
    main()
