"""Data models for the Multimodal Document Extraction System.

Validated output: every extraction returns an Invoice with per-field
confidence scores. Fields below the review threshold are flagged for
a human to check.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    """One row of an invoice/receipt line-items table."""

    description: str
    quantity: float = 1.0
    unit_price: float | None = None
    amount: float
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class FieldResult(BaseModel):
    """A single extracted field with its confidence."""

    value: str | float | None
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool = False


class Invoice(BaseModel):
    """Structured data extracted from one invoice or receipt."""

    document_type: str = "invoice"  # invoice | receipt
    invoice_number: str | None = None
    date: str | None = None  # ISO YYYY-MM-DD when parseable
    vendor: str | None = None
    line_items: list[LineItem] = []
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    currency: str = "USD"


class ExtractionResult(BaseModel):
    """Full output for one document: validated JSON + review flags."""

    source_file: str
    invoice: Invoice
    # confidence per top-level field, 0..1
    field_confidence: dict[str, float] = {}
    # fields whose confidence fell below the review threshold
    needs_review: list[str] = []
    # raw OCR text, kept for audit
    ocr_text: str = ""
    # mean OCR word confidence, 0..1
    ocr_confidence: float = 0.0
    # True when the LLM/vision pass ran on this document
    llm_used: bool = False
    # audit trail of what the LLM pass confirmed / changed / disputed
    llm_notes: list[str] = []
