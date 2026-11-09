# Multimodal Document Extraction System

Extracts structured data (invoice number, date, vendor, line items, total)
from scanned invoices and receipts using **OCR + rule-based extraction +
an LLM / vision-model verification pass**. Output is validated JSON; low-confidence fields are
flagged for human review; field-level accuracy is measured on 30
sample documents.

## Results (30 documents: 20 invoices + 10 receipts)

| Engine | Overall field accuracy |
| ------ | ---------------------- |
| Tesseract 5.3 | **100.0%** |
| RapidOCR (ONNX) | **98.8%** |

Per-field accuracy (RapidOCR): invoice number 100%, date 100%, vendor 100%,
total 100%, subtotal 100%, tax 100%, line items 96.9% (94/97).
The 3 missed line items are genuine OCR errors where both the unit price
and the amount were misread identically (e.g. 9.99 → 666 in both columns),
so the arithmetic cross-check cannot catch them. See `REPORT.md` §5.

## How it works

1. **OCR** — Tesseract or RapidOCR reads the image. Word bounding boxes
   are rebuilt into visual lines (table columns become proper rows).
2. **Extraction** — deterministic patterns pull out invoice number, date,
   vendor, line items, subtotal, tax, total. No API key needed.
3. **Validation** — arithmetic cross-checks repair OCR digit errors:
   qty × unit must equal the amount; total must equal subtotal + tax.
   Corrected values carry lower confidence.
4. **Review flagging** — every field gets a confidence score; anything
   below 0.70 is flagged for a human.
5. **LLM / vision pass** — with `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`
   set, `--refine` sends the OCR text *and the page image* to the model
   (Claude or an OpenAI vision model) with the draft extraction. The reply
   must validate as JSON; a changed value is accepted only if it appears in
   the OCR text; agreement raises confidence, disagreement drops it below
   0.70 so a human reviews it. Every decision is logged in `llm_notes`.
   Without a key the pipeline runs fully offline on rules + OCR.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Tesseract engine needs the tesseract binary; RapidOCR needs none.

# Generate the 30 sample documents (with ground truth)
python -m src.cli generate

# Extract one document -> validated JSON
python -m src.cli extract data/samples/invoice_01.png

# Use Tesseract instead of RapidOCR
python -m src.cli extract data/samples/invoice_01.png --engine tesseract

# Verify with an LLM / vision model (needs ANTHROPIC_API_KEY or OPENAI_API_KEY)
python -m src.cli extract data/samples/invoice_01.png --refine
python -m src.cli extract data/samples/invoice_01.png --refine --provider openai --no-vision

# Measure field-level accuracy on all 30 documents
python -m src.cli evaluate
python -m src.cli evaluate --engine tesseract
python -m src.cli evaluate --refine        # include the LLM/vision pass
```

## Project layout

```
src/
  ocr.py       OCR backends (Tesseract / RapidOCR) + visual line rebuild
  extract.py   field extraction, arithmetic cross-checks, review flagging
  models.py    Pydantic models (validated JSON output)
  generate.py  synthetic invoice/receipt image generator + ground truth
  evaluate.py  field-level accuracy vs ground truth
  llm.py       LLM/vision verification pass (Anthropic / OpenAI, grounded merge)
  cli.py       command-line interface
tests/         30 pytest tests (LLM stage tested with an injected fake model)
data/
  samples/     30 generated invoice/receipt images
  ground_truth.jsonl
results/
  accuracy_report.json            RapidOCR (98.8%)
  accuracy_report_tesseract.json  Tesseract (100.0%)
```

## Sample output

```json
{
  "source_file": "data/samples/invoice_01.png",
  "invoice": {
    "document_type": "invoice",
    "invoice_number": "INV-20260001",
    "date": "2026-05-21",
    "vendor": "Evergreen Paper Co",
    "line_items": [
      {"description": "Desk Lamp LED", "quantity": 1, "unit_price": 32.0, "amount": 32.0}
    ],
    "subtotal": 360.95,
    "tax": 28.88,
    "total": 389.83
  },
  "field_confidence": {"invoice_number": 1.0, "date": 1.0, "vendor": 0.8, "total": 1.0},
  "needs_review": [],
  "ocr_confidence": 0.986
}
```

## Tests

```bash
python -m pytest tests/ -q   # 30 passed
```

## Notes

- The headline accuracy numbers are from the offline OCR + rules pipeline.
  The LLM stage is implemented and unit-tested with a fake model, but its
  accuracy has **not** been measured here (no API key in the build
  environment) — run `evaluate --refine` with your key to measure it.
- Review flags: receipts have no subtotal/tax block, so those are no longer
  flagged when absent (this was a false-flag bug). `evaluate` now also
  reports how many wrong fields were flagged and line-item precision.
  `results/accuracy_report.json` (RapidOCR) predates this change: its
  accuracy numbers are unchanged but its flag counts are stale; the
  Tesseract report is current.

- Sample documents are synthetic (generated with PIL) so the ground truth
  is exact and the set is redistributable. The same pipeline runs on real
  scans — pass any image to `extract`.
- The framework is reusable: `extract_document()` takes any OCR result,
  and the LLM pass takes any OpenAI-compatible endpoint.
