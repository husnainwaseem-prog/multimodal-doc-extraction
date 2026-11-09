"""Tests for the LLM/vision verification stage (no network: llm_fn is injected)."""

from __future__ import annotations

import json

import pytest

from src.extract import extract_document
from src.llm import parse_llm_json, refine_with_llm
from tests.test_extraction import SAMPLE_INVOICE, _ocr


def _base():
    return extract_document("x.png", _ocr(SAMPLE_INVOICE))


def _reply(**over):
    inv = _base().invoice.model_dump()
    inv.update(over)
    return json.dumps(inv)


def test_agreement_confirms_and_keeps_no_flags():
    res = refine_with_llm(_base(), llm_fn=lambda p, i: _reply())
    assert res.llm_used and res.needs_review == []
    assert res.field_confidence["total"] >= 0.95


def test_grounded_disagreement_is_adopted_but_flagged():
    base = _base()
    wrong_date = _reply(date="2026-03-14")
    base.invoice.date = "2026-03-11"  # pretend rules misread the date
    res = refine_with_llm(base, llm_fn=lambda p, i: wrong_date)
    assert res.invoice.date == "2026-03-14"
    assert "date" in res.needs_review


def test_hallucinated_value_rejected_and_flagged():
    res = refine_with_llm(_base(), llm_fn=lambda p, i: _reply(total=999.99))
    assert res.invoice.total == 67.50  # unchanged
    assert "total" in res.needs_review


def test_invalid_json_keeps_rule_result():
    res = refine_with_llm(_base(), llm_fn=lambda p, i: "not json")
    assert not res.llm_used
    assert any("failed" in n for n in res.llm_notes)


def test_markdown_fenced_json_parses():
    assert parse_llm_json("```json\n" + _reply() + "\n```").vendor == "Acme Office Supplies"


def test_no_key_returns_unchanged(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    res = refine_with_llm(_base())
    assert not res.llm_used and any("skipped" in n for n in res.llm_notes)


def test_receipt_missing_subtotal_tax_not_flagged():
    text = "Shop One\nReceipt No: RCP-50008\n2026-02-08\nWidget 1 x 5.00 5.00\nTotal 5.00"
    res = extract_document("r.png", _ocr(text), doc_type="receipt")
    assert "subtotal" not in res.needs_review and "tax" not in res.needs_review
