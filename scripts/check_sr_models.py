#!/usr/bin/env python3
"""Check SR model configuration without running reconstruction."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sr_3dgs.step2_super_resolution import SuperResolutionProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_model", default="real-esrgan",
                        choices=["real-esrgan", "dat", "supir", "basicvsr++"])
    parser.add_argument("--sr_scale", type=int, default=2, choices=[1, 2, 4, 8])
    parser.add_argument("--sr_device", default="cuda")
    parser.add_argument("--model_path", default="",
                        help="Optional local weight file or URL for supported models")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if the preflight reports a problem")
    args = parser.parse_args()

    kwargs = {"model_path": args.model_path} if args.model_path else {}
    processor = SuperResolutionProcessor(
        image_dir=".",
        output_dir=".",
        sr_model_name=args.sr_model,
        scale=args.sr_scale,
        device=args.sr_device,
        model_kwargs=kwargs,
        mode="model",
    )
    report = processor._model_preflight()
    print(json.dumps(report, indent=2))
    if args.strict and report.get("ok") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
