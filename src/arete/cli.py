"""Command-line smoke example; never starts an experiment campaign."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Arete beam-hopping research code")
    parser.add_argument("command", choices=["demo", "config"])
    args = parser.parse_args()
    if args.command == "demo":
        from .demo import run_demo
        output = run_demo()
    else:
        from .config import paper_config
        output = paper_config()
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
