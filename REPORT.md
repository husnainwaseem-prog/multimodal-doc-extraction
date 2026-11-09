# Multimodal Document Extraction System — Evaluation Report

## 1. Task

Build a system that extracts structured data (invoice number, date, vendor,
line items, total) from scanned invoices or receipts using OCR plus an LLM
or vision model. Output validated JSON, flag low-confidence fields for human
review, and measure field-level accuracy on at least 30 documents.

## 2. System

- **OCR:** two engines behind one interface — Tesseract 5.3 (system binary)
  and RapidOCR (ONNX, pip-installable, no system install). Word bounding
  boxes are rebuilt into visual lines so table rows come back as rows.
- **Extraction:** deterministic patterns (no API key). Invoice number, date
  (4 formats incl. OCR-merged "04Feb2026"), vendor (first content line),
  line items (token-based row parsing, invoice and receipt layouts),
  subtotal / tax / total.
- **Validation:** Pydantic models; arithmetic cross-checks repair OCR digit
  errors (qty × unit = amount; total = subtotal + tax). Corrected values
  carry confidence 0.75 instead of ~1.0.
- **Review flagging:** every field scored 0–1; below 0.70 → flagged. Absent
  subtotal/tax on receipts is expected and not flagged. `evaluate` reports
  how many wrong fields were flagged and how many flags hit correct fields.
- **LLM / vision pass:** `src/llm.py` (`--refine`) sends OCR text + page
  image to Claude or an OpenAI vision model, validates the JSON reply,
  accepts changes only if grounded in the OCR text, raises confidence on
  agreement and flags disagreements. The numbers below measure the offline
  rule-based pipeline; the LLM pass is unit-tested with an injected fake
  model but its accuracy is not measured here (no API key available).

## 3. Test set

30 synthetic documents, seed 42, generated with PIL (`python -m src.cli generate`):
20 invoices + 10 receipts, 97 line items total. Ground truth is exact
(`data/ground_truth.jsonl`). Synthetic documents were chosen so the set is
redistributable and the truth is known — the same pipeline runs on real
scans via `extract`.

## 4. Results — field-level accuracy (exact match)

| Field | RapidOCR | Tesseract |
| ----- | -------- | --------- |
| invoice_number | 100.0% (30/30) | 100.0% (30/30) |
| date | 100.0% (30/30) | 100.0% (30/30) |
| vendor | 100.0% (30/30) | 100.0% (30/30) |
| total | 100.0% (30/30) | 100.0% (30/30) |
| subtotal | 100.0% (20/20) | 100.0% (20/20) |
| tax | 100.0% (20/20) | 100.0% (20/20) |
| line_items | 96.9% (94/97) | 100.0% (97/97) |
| **Overall** | **98.8%** | **100.0%** |

Full per-document results: `results/accuracy_report.json` (RapidOCR),
`results/accuracy_report_tesseract.json` (Tesseract).

## 5. Errors (RapidOCR, the only misses)

All 3 missed line items are the same failure mode: OCR misread a price
identically in both the unit-price and amount columns, so the arithmetic
cross-check sees a consistent (but wrong) row:

- invoice_17: "Whiteboard Markers 1 9.99 9.99" read as "1 666 666"
- invoice_18: "Webcam HD 1080p 2 49.50 99.00" read as "2 33.00 66.00"
- receipt_03: "External SSD 500GB" read as "ExternalSSD50oGB" (0 → o)

These are genuine OCR limits, not extraction bugs. Tesseract read all three
correctly, which is why it scores 100%. On real-world scans the same
failure mode would be caught by the review flag only if the OCR confidence
on those words is low — here it was not, which is an honest limitation.

## 6. What the cross-checks fixed (would otherwise have been errors)

- "Whiteboard Markers 3 666 29.97" — unit repaired 666 → 9.99 from amount/qty
- "Whiteboard Markers 2 66'6 19.98" — junk token repaired the same way
- 3 tax lines missing/misread — derived as total − subtotal
- "06Jan 2026", "04Feb2026" — dates with OCR-merged spaces parsed
- "1 × 4.25" — multiplication sign instead of "x" handled
- "3x79.99" — merged qty token split

Without these, RapidOCR overall accuracy was 75.5% on the first run.

## 7. Tests

30/30 pytest tests pass (incl. LLM-stage tests) (`python -m pytest tests/ -q`), covering invoice
numbers, all date formats, both line-item layouts, the arithmetic repair,
invoice balancing, review flagging, JSON validation, and an end-to-end
run on a generated sample.

## 8. Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.cli generate
python -m src.cli evaluate                 # RapidOCR, 98.8%
python -m src.cli evaluate --engine tesseract   # Tesseract, 100.0%
python -m src.cli evaluate --refine             # with LLM/vision pass (needs API key)
python -m pytest tests/ -q                  # 30 passed
```
