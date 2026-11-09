"""Field-level accuracy evaluation.

Compares each extraction against ground truth field by field and
reports exact-match accuracy per field type, plus an overall score.
A field counts as correct only when the extracted value equals the
ground-truth value (money compared to the cent, dates compared as
ISO strings, text compared case-insensitively after stripping).
"""

from __future__ import annotations

import json
from pathlib import Path

from datetime import datetime

from .extract import extract_document
from .models import ExtractionResult
from .ocr import ocr_image

FIELDS = ["invoice_number", "date", "vendor", "total", "subtotal", "tax"]

_DATE_FMTS = ["%Y-%m-%d", "%m/%d/%Y", "%d %b %Y", "%B %d, %Y", "%d %B %Y"]


def _to_iso(s: str) -> str:
    """Normalise any rendered date to ISO YYYY-MM-DD."""
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(s.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return s.strip()


def _norm(v, field: str = "") -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.2f}"
    s = str(v).strip().lower()
    if field == "date":
        s = _to_iso(s)
    # OCR word segmentation is unreliable ("BluePeak Distributors"
    # can come back as "BluePeakDistributors") — compare text fields
    # with whitespace removed; the words themselves must all match.
    if field in ("vendor", "description"):
        s = "".join(s.split())
    return s


def field_correct(field: str, got, want) -> bool:
    return _norm(got, field) == _norm(want, field) and _norm(want, field) != ""


def line_items_correct(got: list, want: list) -> tuple[int, int]:
    """Return (correct_items, total_items) — an item is correct when
    description, quantity and amount all match."""
    correct = 0
    for w in want:
        for g in got:
            if (_norm(g.description, "description") == _norm(w["description"], "description")
                    and float(g.quantity) == float(w["quantity"])
                    and abs(float(g.amount) - float(w["amount"])) < 0.005):
                correct += 1
                break
    return correct, len(want)


def evaluate(ground_truth_path: str, samples_dir: str, engine: str = "rapidocr",
             out_json: str | None = None, refine: bool = False,
             provider: str | None = None, model: str | None = None) -> dict:
    """Run extraction on every sample and score against ground truth."""
    gt_rows = [json.loads(ln) for ln in open(ground_truth_path)]
    per_field: dict[str, dict[str, int]] = {f: {"correct": 0, "total": 0} for f in FIELDS}
    per_field["line_items"] = {"correct": 0, "total": 0}
    flagged = 0
    err_total = err_flagged = false_flags = 0
    spurious = produced = 0
    results: list[dict] = []

    for row in gt_rows:
        img = str(Path(samples_dir) / row["file"])
        ocr = ocr_image(img, engine=engine)
        res = extract_document(img, ocr, doc_type=row["document_type"])
        if refine:
            from .llm import refine_with_llm
            res = refine_with_llm(res, image_path=img, provider=provider, model=model)
        inv = res.invoice

        got = {
            "invoice_number": inv.invoice_number,
            "date": inv.date,
            "vendor": inv.vendor,
            "total": inv.total,
            "subtotal": inv.subtotal,
            "tax": inv.tax,
        }
        row_score: dict[str, bool] = {}
        for f in FIELDS:
            want = row.get(f)
            if want is None:
                continue
            per_field[f]["total"] += 1
            ok = field_correct(f, got[f], want)
            per_field[f]["correct"] += int(ok)
            row_score[f] = ok
        li_ok, li_n = line_items_correct(inv.line_items, row["line_items"])
        per_field["line_items"]["correct"] += li_ok
        per_field["line_items"]["total"] += li_n
        row_score["line_items"] = li_ok == li_n
        produced += len(inv.line_items)
        spurious += max(0, len(inv.line_items) - li_ok)
        # does the review flag actually catch the mistakes?
        flags = set(res.needs_review)
        for f, ok in row_score.items():
            is_flagged = f in flags or (f == "line_items" and any(x.startswith("line_items") for x in flags))
            if not ok:
                err_total += 1
                err_flagged += int(is_flagged)
            elif is_flagged:
                false_flags += 1
        if res.needs_review:
            flagged += 1
        results.append({
            "file": row["file"],
            "fields": row_score,
            "needs_review": res.needs_review,
            "ocr_confidence": round(res.ocr_confidence, 3),
        })

    field_acc = {
        f: (round(c["correct"] / c["total"], 4) if c["total"] else None)
        for f, c in per_field.items()
    }
    totals = [c["total"] for c in per_field.values()]
    corrects = [c["correct"] for c in per_field.values()]
    overall = round(sum(corrects) / sum(totals), 4) if sum(totals) else 0.0

    report = {
        "n_documents": len(gt_rows),
        "engine": engine,
        "overall_field_accuracy": overall,
        "field_accuracy": field_acc,
        "field_counts": {f: c for f, c in per_field.items()},
        "documents_flagged_for_review": flagged,
        "llm_refinement": refine,
        "review_flag_quality": {
            "errors_total": err_total,
            "errors_flagged": err_flagged,
            "false_flags": false_flags,
        },
        "line_item_precision": round(1 - spurious / produced, 4) if produced else 1.0,
        "per_document": results,
    }
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(report, indent=2))
    return report
