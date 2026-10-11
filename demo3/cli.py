"""Headless version: python -m demo3.cli input.wav"""
import argparse
import json
from .ai_pipeline import process_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = process_file(args.input, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
