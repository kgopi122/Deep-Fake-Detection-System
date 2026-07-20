from __future__ import annotations

import argparse

from .config import DetectorConfig
from .pipeline import detect_image


def main() -> None:
    p = argparse.ArgumentParser(prog="veritrue-detect")
    p.add_argument("image", help="Path to an image")
    p.add_argument("--json", action="store_true", help="Print JSON output")
    p.add_argument("--exiftool", default=None, help="Path to exiftool.exe (optional)")
    p.add_argument("--c2patool", default=None, help="Path to c2patool (optional)")
    args = p.parse_args()

    cfg = DetectorConfig(exiftool_path=args.exiftool, c2patool_path=args.c2patool)
    report = detect_image(args.image, cfg)

    if args.json:
        print(report.to_json())
    else:
        print(f"AI-likelihood: {report.ai_likelihood_0_100}/100")
        print(f"Verdict: {report.verdict}")
        for line in report.explanation:
            print("-", line)


if __name__ == "__main__":
    main()
