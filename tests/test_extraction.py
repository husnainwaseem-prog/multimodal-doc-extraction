"""Tests for the Multimodal Document Extraction System.

Run: python -m pytest tests/ -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluate import _norm, _to_iso, field_correct, line_items_correct
from src.extract import (
    extract_date,
    extract_document,
    extract_invoice_number,
    extract_line_items,
    extract_money_field,
    TOTAL_RE,
)
from src.models import ExtractionResult, Invoice, LineItem
from src.ocr import OCRResult


def _ocr(text: str) -> OCRResult:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    words = [(w, 0.99) for ln in lines for w in ln.split()]
    return OCRResult(text=text, words=words, lines=lines)


SAMPLE_INVOICE = """Acme Office Supplies
INVOICE
Invoice No: INV-20260001
Date: 2026-03-14
Description Qty Unit Price Amount
Wireless Mouse 2 25.00 50.00
USB-C Cable 1m 1 12.50 12.50
Subtotal 62.50
Tax 5.00
Total 67.50
Thank you for your business."""


# --- invoice number ---------------------------------------------------

def test_invoice_number():
    ocr = _ocr(SAMPLE_INVOICE)
    r = extract_invoice_number(ocr.lines, ocr)
    assert r.value == "INV-20260001"
    assert r.confidence > 0.9


def test_invoice_number_not_from_header():
    # "INVOICE" alone must not produce a number (regression: "OICE")
    ocr = _ocr("INVOICE\nDate: 2026-03-14")
    r = extract_invoice_number(ocr.lines, ocr)
    assert r.value is None


# --- dates ------------------------------------------------------------

@pytest.mark.parametrize("raw,iso", [
    ("Date: 2026-03-14", "2026-03-14"),
    ("Date: 03/14/2026", "2026-03-14"),
    ("Date: 14 Mar 2026", "2026-03-14"),
    ("Date: 06Jan 2026", "2026-01-06"),   # OCR merges day+month
    ("Date: 04Feb2026", "2026-02-04"),    # OCR merges everything
    ("Date: March 14, 2026", "2026-03-14"),
])
def test_date_formats(raw, iso):
    ocr = _ocr(raw)
    r = extract_date(ocr.lines, ocr)
    assert r.value == iso


def test_to_iso_normalises_ground_truth():
    assert _to_iso("03/14/2026") == "2026-03-14"
    assert _to_iso("14 Mar 2026") == "2026-03-14"


# --- money fields ------------------------------------------------------

def test_total_subtotal_tax():
    ocr = _ocr(SAMPLE_INVOICE)
    assert extract_money_field(ocr.lines, TOTAL_RE, ocr).value == 67.50


# --- line items --------------------------------------------------------

def test_line_items_invoice_format():
    ocr = _ocr(SAMPLE_INVOICE)
    items = extract_line_items(ocr.lines, ocr)
    assert len(items) == 2
    assert items[0].description == "Wireless Mouse"
    assert items[0].quantity == 2
    assert items[0].unit_price == 25.00
    assert items[0].amount == 50.00


def test_line_items_receipt_format():
    text = "Shop\nRECEIPT\nReceipt No: RCP-50001\nDate: 2026-02-08\nWireless Mouse 1 x 25.00 25.00\nTotal 25.00"
    ocr = _ocr(text)
    items = extract_line_items(ocr.lines, ocr)
    assert len(items) == 1
    assert items[0].description == "Wireless Mouse"
    assert items[0].amount == 25.00


def test_line_items_cross_multiplication_sign():
    # OCR sometimes reads "x" as "×"
    text = "Shop\nRECEIPT\nNotebook A5 1 × 4.25 4.25\nTotal 4.25"
    ocr = _ocr(text)
    items = extract_line_items(ocr.lines, ocr)
    assert len(items) == 1
    assert items[0].description == "Notebook A5"


def test_line_items_ocr_digit_repair():
    # unit price misread as 666, amount correct -> recompute unit
    text = "Shop\nINVOICE\nWhiteboard Markers 3 666 29.97\nTotal 29.97"
    ocr = _ocr(text)
    items = extract_line_items(ocr.lines, ocr)
    assert len(items) == 1
    assert items[0].unit_price == 9.99
    assert items[0].confidence < 0.9  # corrected values carry lower confidence


def test_line_items_inconsistent_row_skipped():
    # qty x unit != amount and no clean fix -> not a real row
    text = "Shop\nINVOICE\nRandom Words 2 10.00 99.99\nTotal 99.99"
    ocr = _ocr(text)
    items = extract_line_items(ocr.lines, ocr)
    assert items == []


# --- full document ------------------------------------------------------

def test_full_extraction():
    ocr = _ocr(SAMPLE_INVOICE)
    res = extract_document("invoice_01.png", ocr)
    assert res.invoice.invoice_number == "INV-20260001"
    assert res.invoice.date == "2026-03-14"
    assert res.invoice.vendor == "Acme Office Supplies"
    assert res.invoice.total == 67.50
    assert res.invoice.subtotal == 62.50
    assert res.invoice.tax == 5.00
    assert len(res.invoice.line_items) == 2
    assert res.needs_review == []


def test_tax_derived_from_balancing():
    # tax line missing entirely -> total - subtotal
    text = "Shop\nINVOICE\nInvoice No: INV-1\nDate: 2026-01-01\nItem A 1 10.00 10.00\nSubtotal 10.00\nTotal 10.80"
    ocr = _ocr(text)
    res = extract_document("x.png", ocr)
    assert res.invoice.tax == 0.80


def test_low_confidence_fields_flagged():
    # a document with no total -> total flagged for review
    text = "Shop\nINVOICE\nInvoice No: INV-1\nDate: 2026-01-01"
    ocr = _ocr(text)
    res = extract_document("x.png", ocr)
    assert "total" in res.needs_review


def test_validated_json_roundtrip():
    ocr = _ocr(SAMPLE_INVOICE)
    res = extract_document("invoice_01.png", ocr)
    data = json.loads(res.model_dump_json())
    assert data["invoice"]["total"] == 67.50
    assert data["invoice"]["invoice_number"] == "INV-20260001"
    # re-validate from JSON
    ExtractionResult.model_validate(data)


# --- evaluation helpers -------------------------------------------------

def test_field_correct_normalises_dates():
    assert field_correct("date", "2026-03-14", "03/14/2026")
    assert field_correct("date", "2026-03-14", "14 Mar 2026")


def test_field_correct_ignores_ocr_spacing():
    assert field_correct("vendor", "BluePeakDistributors", "BluePeak Distributors")
    assert not field_correct("vendor", "Acme Corp", "Beta Corp")


def test_line_items_correct():
    got = [LineItem(description="Wireless Mouse", quantity=2, unit_price=25.0, amount=50.0)]
    want = [{"description": "Wireless Mouse", "quantity": 2, "unit_price": 25.0, "amount": 50.0}]
    ok, n = line_items_correct(got, want)
    assert (ok, n) == (1, 1)


# --- end-to-end on a real generated sample ------------------------------

def test_end_to_end_one_sample():
    from src.ocr import available_engines, ocr_image

    samples = sorted(Path("data/samples").glob("invoice_*.png"))
    assert samples, "run: python -m src.cli generate"
    engines = available_engines()
    if not engines:
        pytest.skip("no OCR engine installed")
    ocr = ocr_image(str(samples[0]), engine=engines[0])
    res = extract_document(str(samples[0]), ocr)
    assert res.invoice.invoice_number is not None
    assert res.invoice.total is not None
    assert len(res.invoice.line_items) >= 1
