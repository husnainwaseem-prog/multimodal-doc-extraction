"""CLI for the Multimodal Document Extraction System.

Usage:
    python -m src.cli generate [--n-invoices 20] [--n-receipts 10]
    python -m src.cli extract <image> [--engine rapidocr|tesseract] [--out results/out.json]
    python -m src.cli evaluate [--engine rapidocr|tesseract]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import evaluate  # noqa: E402
from src.extract import extract_document  # noqa: E402
from src.generate import generate_dataset  # noqa: E402
from src.ocr import available_engines, ocr_image  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Multimodal Document Extraction System")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate synthetic sample documents")
    g.add_argument("--n-invoices", type=int, default=20)
    g.add_argument("--n-receipts", type=int, default=10)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out", default="data/samples")

    e = sub.add_parser("extract", help="Extract structured data from one image")
    e.add_argument("image")
    e.add_argument("--engine", default="rapidocr", choices=["rapidocr", "tesseract"])
    e.add_argument("--refine", action="store_true",
                   help="Verify with an LLM/vision model (needs ANTHROPIC_API_KEY or OPENAI_API_KEY)")
    e.add_argument("--provider", choices=["anthropic", "openai"], default=None)
    e.add_argument("--model", default=None)
    e.add_argument("--no-vision", action="store_true", help="send OCR text only, not the image")
    e.add_argument("--out", default=None)

    v = sub.add_parser("evaluate", help="Measure field-level accuracy on the sample set")
    v.add_argument("--engine", default="rapidocr", choices=["rapidocr", "tesseract"])
    v.add_argument("--gt", default="data/ground_truth.jsonl")
    v.add_argument("--samples", default="data/samples")
    v.add_argument("--out", default="results/accuracy_report.json")
    v.add_argument("--refine", action="store_true", help="include the LLM/vision pass")
    v.add_argument("--provider", choices=["anthropic", "openai"], default=None)
    v.add_argument("--model", default=None)

    args = p.parse_args()

    if args.cmd == "generate":
        truth = generate_dataset(args.n_invoices, args.n_receipts, args.seed, args.out)
        print(f"Generated {len(truth)} documents in {args.out}/")
        print(f"Ground truth written to data/ground_truth.jsonl")
    elif args.cmd == "extract":
        ocr = ocr_image(args.image, engine=args.engine)
        res = extract_document(args.image, ocr)
        if args.refine:
            from src.llm import refine_with_llm
            res = refine_with_llm(res, image_path=None if args.no_vision else args.image,
                                  provider=args.provider, model=args.model)
            for n in res.llm_notes:
                print(f"[llm] {n}")
        out = res.model_dump_json(indent=2, exclude={"ocr_text"})
        if args.out:
            Path(args.out).write_text(out)
            print(f"Wrote {args.out}")
        else:
            print(out)
        if res.needs_review:
            print(f"\nFlagged for human review: {', '.join(res.needs_review)}")
    elif args.cmd == "evaluate":
        print(f"OCR engines available: {available_engines()}")
        report = evaluate(args.gt, args.samples, engine=args.engine, out_json=args.out,
                          refine=args.refine, provider=args.provider, model=args.model)
        print(f"\nDocuments: {report['n_documents']}  |  Engine: {report['engine']}")
        print(f"Overall field accuracy: {report['overall_field_accuracy']:.1%}")
        for f, acc in report["field_accuracy"].items():
            c = report["field_counts"][f]
            print(f"  {f:15s} {acc:.1%}  ({c['correct']}/{c['total']})")
        print(f"Documents flagged for review: {report['documents_flagged_for_review']}")
        rv = report["review_flag_quality"]
        print(f"Review flags: {rv['errors_flagged']}/{rv['errors_total']} wrong fields were flagged; "
              f"{rv['false_flags']} flags on correct fields")
        print(f"Line-item precision (no spurious rows): {report['line_item_precision']:.1%}")
        print(f"\nFull report: {args.out}")


if __name__ == "__main__":
    main()
