"""Field extraction from OCR text.

Approach: deterministic pattern matching on the OCR output — invoice
numbers, dates, money amounts and line-item rows all have strong
surface patterns, so regex + layout heuristics extract them without
needing an LLM call. An optional LLM/VLM pass (see llm.py) can refine
ambiguous documents; the rule-based pass is the baseline and is what
the accuracy report measures.

Every field gets a confidence score in [0, 1]:
- 1.0  exact pattern match on clean OCR words
- 0.8  pattern match, some OCR words low-confidence
- 0.5  fallback heuristic (e.g. first line as vendor)
- 0.0  not found
Fields below REVIEW_THRESHOLD (0.70) are flagged for human review.
"""

from __future__ import annotations

import re
from datetime import datetime

from .models import ExtractionResult, FieldResult, Invoice, LineItem
from .ocr import OCRResult

REVIEW_THRESHOLD = 0.70

# --- patterns ---------------------------------------------------------

MONEY_RE = re.compile(r"[$€£]?\s*(\d{1,3}(?:,\d{3})*\.\d{2})")
MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}

DATE_PATTERNS = [
    # 2026-03-14 / 2026/03/14
    (re.compile(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b"), "ymd"),
    # 03/14/2026
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b"), "mdy"),
    # 14 Mar 2026 / 06Jan 2026 / 04Feb2026 / March 14, 2026 (spaces optional — OCR merges)
    (re.compile(r"\b(\d{1,2})\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*(20\d{2})\b", re.I), "dmy"),
    (re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(20\d{2})\b", re.I), "mdy_name"),
]

INVOICE_NO_RE = re.compile(
    r"\b(?:invoice\s*(?:no\.?|number|#)?|inv\.?\s*(?:no\.?|#)?|receipt\s*(?:no\.?|#)?)\s*[:#]?\s*([A-Za-z]{2,4}[-/]?\d{3,}|[A-Z]{2,}\d+[-/]?\d*)\b",
    re.I,
)

TOTAL_RE = re.compile(r"\b(?:grand\s+)?total\b[^0-9$€£]*[$€£]?\s*(\d{1,3}(?:,\d{3})*\.\d{2})", re.I)
SUBTOTAL_RE = re.compile(r"\bsubtotal\b[^0-9$€£]*[$€£]?\s*(\d{1,3}(?:,\d{3})*\.\d{2})", re.I)
TAX_RE = re.compile(r"\b(?:tax|vat)\b[^0-9$€£]*[$€£]?\s*(\d{1,3}(?:,\d{3})*\.\d{2})", re.I)

# line-item row: description ... qty ... unit price ... amount
# e.g. "Wireless Mouse  2  25.00  50.00"
ITEM_RE = re.compile(
    r"^(.+?)\s+(\d+(?:\.\d+)?)\s+[$€£]?(\d{1,3}(?:,\d{3})*\.\d{2})\s+[$€£]?(\d{1,3}(?:,\d{3})*\.\d{2})\s*$"
)
# receipt-style row: description ... qty x unit price ... amount
# e.g. "Wireless Mouse  1 x 25.00  25.00"
ITEM_X_RE = re.compile(
    r"^(.+?)\s+(\d+(?:\.\d+)?)\s*x\s*[$€£]?(\d{1,3}(?:,\d{3})*\.\d{2})\s+[$€£]?(\d{1,3}(?:,\d{3})*\.\d{2})\s*$"
)

SKIP_LINE_WORDS = {
    "invoice", "receipt", "total", "subtotal", "tax", "vat", "balance",
    "amount", "due", "thank", "payment", "date", "bill", "ship",
}


def _money(s: str) -> float:
    return float(s.replace(",", ""))


def _word_conf(text: str, ocr: OCRResult) -> float:
    """Mean OCR confidence of the words appearing in `text`."""
    confs = [c for w, c in ocr.words if w.lower() in text.lower()]
    return sum(confs) / len(confs) if confs else 0.8


def extract_invoice_number(lines: list[str], ocr: OCRResult) -> FieldResult:
    for line in lines:
        m = INVOICE_NO_RE.search(line)
        if m:
            val = m.group(1).strip().strip(",")
            return FieldResult(value=val, confidence=_word_conf(val, ocr))
    return FieldResult(value=None, confidence=0.0)


def extract_date(lines: list[str], ocr: OCRResult) -> FieldResult:
    for line in lines:
        if "total" in line.lower() or "due" in line.lower():
            continue
        for pat, kind in DATE_PATTERNS:
            m = pat.search(line)
            if m:
                raw = m.group(0)
                iso = None
                try:
                    if kind == "ymd":
                        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    elif kind == "mdy":
                        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    elif kind == "dmy":
                        d, mo, y = int(m.group(1)), MONTHS[m.group(2).lower()[:3]], int(m.group(3))
                    else:  # mdy_name
                        mo, d, y = MONTHS[m.group(1).lower()[:3]], int(m.group(2)), int(m.group(3))
                    iso = datetime(y, mo, d).date().isoformat()
                except (ValueError, KeyError):
                    iso = raw
                return FieldResult(value=iso, confidence=_word_conf(raw, ocr))
    return FieldResult(value=None, confidence=0.0)


def extract_vendor(lines: list[str], ocr: OCRResult) -> FieldResult:
    """Vendor is usually the first non-empty line (company name)."""
    for line in lines:
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if any(w in low for w in ("invoice", "receipt", "tax id")):
            continue
        if MONEY_RE.search(s) or re.search(r"\d{4}", s):
            continue
        return FieldResult(value=s, confidence=0.8)
    return FieldResult(value=None, confidence=0.0)


def extract_money_field(lines: list[str], pattern: re.Pattern, ocr: OCRResult) -> FieldResult:
    for line in lines:
        m = pattern.search(line)
        if m:
            return FieldResult(value=_money(m.group(1)), confidence=_word_conf(m.group(1), ocr))
    return FieldResult(value=None, confidence=0.0)


def _clean_money(tok: str) -> float | None:
    """Parse a money token, tolerating OCR junk (e.g. 66'6 -> 666)."""
    s = re.sub(r"[^0-9.,]", "", tok).replace(",", "")
    if not s or s.count(".") > 1:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _split_qty_token(tokens: list[str]) -> list[str]:
    """Split merged qty tokens OCR produces: '3x79.99' -> ['3', 'x', '79.99']."""
    out: list[str] = []
    for tok in tokens:
        m = re.match(r"^(\d+)[x×](\d.*)$", tok)
        if m:
            out.extend([m.group(1), "x", m.group(2)])
        else:
            out.append(tok)
    return out


def extract_line_items(lines: list[str], ocr: OCRResult) -> list[LineItem]:
    """Token-based row parsing.

    A row is: <description...> <qty> <unit price> <amount>.
    OCR can mangle the unit price ("9.99" -> "666" or "66'6"), so we
    parse by position — last token is the amount, the one before it
    the unit price, the one before that the quantity — and then use
    the arithmetic identity qty x unit = amount to repair the unit
    price when OCR misread it.
    """
    items: list[LineItem] = []
    in_table = False
    for line in lines:
        low = line.lower().strip()
        if re.search(r"\b(description|item|qty|quantity|price|amount)\b", low) and not in_table:
            in_table = True
            continue
        if low.startswith(("subtotal", "total", "tax", "vat", "balance", "thank")):
            in_table = False
            continue
        tokens = _split_qty_token(line.strip().split())
        if len(tokens) < 4:
            continue
        amount = _clean_money(tokens[-1])
        unit = _clean_money(tokens[-2])
        # receipt rows put an "x" (or "×") between qty and unit price:
        # "Wireless Mouse 1 x 25.00 25.00"
        if tokens[-3].lower() in ("x", "×") and len(tokens) >= 5:
            qty_tok, desc_toks = tokens[-4], tokens[:-4]
        else:
            qty_tok, desc_toks = tokens[-3], tokens[:-3]
        try:
            qty = float(re.sub(r"[^0-9.]", "", qty_tok))
        except ValueError:
            continue
        if amount is None or unit is None or qty <= 0:
            continue
        # qty token must look like a small integer (1-99), not a price
        if qty > 99 or ("." in qty_tok and len(qty_tok.split(".")[1]) == 2):
            continue
        desc = " ".join(desc_toks).strip()
        if not desc or desc.lower() in SKIP_LINE_WORDS or len(desc) < 2:
            continue
        # skip header-like rows and the totals block
        if re.search(r"\b(description|qty|price|amount|total|subtotal|tax)\b", desc.lower()):
            continue
        conf = _word_conf(desc, ocr)
        # Arithmetic cross-check: qty x unit must equal amount.
        # Only repair the unit price when the OCR token was clearly
        # dirty (no decimal point, or junk characters like 66'6) —
        # a clean price that does not multiply out means the row is
        # genuinely inconsistent, so skip it instead.
        unit_dirty = not re.match(r"^[$€£]?\d{1,3}(?:,\d{3})*\.\d{2}$", tokens[-2])
        if abs(qty * unit - amount) > 0.01:
            if unit_dirty:
                fixed = round(amount / qty, 2)
                if abs(qty * fixed - amount) < 0.005:
                    unit = fixed
                    conf = min(conf, 0.75)  # corrected value, slightly lower confidence
                else:
                    continue
            else:
                continue  # clean price, row does not add up — not a real line item
        items.append(LineItem(
            description=desc,
            quantity=qty,
            unit_price=unit,
            amount=amount,
            confidence=conf,
        ))
    return items


def extract_document(image_path: str, ocr: OCRResult, doc_type: str = "invoice") -> ExtractionResult:
    """Run the full extraction pipeline on one OCR result."""
    lines = [ln.strip() for ln in ocr.lines if ln.strip()]

    inv_no = extract_invoice_number(lines, ocr)
    date = extract_date(lines, ocr)
    vendor = extract_vendor(lines, ocr)
    total = extract_money_field(lines, TOTAL_RE, ocr)
    subtotal = extract_money_field(lines, SUBTOTAL_RE, ocr)
    tax = extract_money_field(lines, TAX_RE, ocr)
    items = extract_line_items(lines, ocr)

    # vendor confidence is a heuristic fallback: 0.8 if found
    if vendor.value is not None and vendor.confidence == 0.0:
        vendor.confidence = 0.8

    invoice = Invoice(
        document_type=doc_type,
        invoice_number=inv_no.value if isinstance(inv_no.value, str) else None,
        date=date.value if isinstance(date.value, str) else None,
        vendor=vendor.value if isinstance(vendor.value, str) else None,
        line_items=items,
        subtotal=subtotal.value if isinstance(subtotal.value, float) else None,
        tax=tax.value if isinstance(tax.value, float) else None,
        total=total.value if isinstance(total.value, float) else None,
    )

    # Invoice balancing: total = subtotal + tax. If OCR dropped or
    # misread the tax line but total and subtotal are solid, derive it.
    if invoice.total is not None and invoice.subtotal is not None:
        implied_tax = round(invoice.total - invoice.subtotal, 2)
        if invoice.tax is None:
            invoice.tax = implied_tax
            tax.confidence = 0.75
        elif abs(invoice.tax - implied_tax) > 0.01:
            invoice.tax = implied_tax
            tax.confidence = min(tax.confidence, 0.75)

    field_conf = {
        "invoice_number": inv_no.confidence,
        "date": date.confidence,
        "vendor": vendor.confidence,
        "total": total.confidence,
        "subtotal": subtotal.confidence,
        "tax": tax.confidence,
    }
    # Receipts normally have no subtotal/tax block, so an absent value is
    # expected there and must not be flagged. Invoices need every field.
    optional = {"subtotal", "tax"} if doc_type == "receipt" else set()
    needs_review = [
        f for f, c in field_conf.items()
        if c < REVIEW_THRESHOLD and not (f in optional and getattr(invoice, f) is None)
    ]
    if not items:
        needs_review.append("line_items")
    # line items with low confidence also need review
    for i, it in enumerate(items):
        if it.confidence < REVIEW_THRESHOLD:
            needs_review.append(f"line_items[{i}]")

    return ExtractionResult(
        source_file=image_path,
        invoice=invoice,
        field_confidence=field_conf,
        needs_review=needs_review,
        ocr_text=ocr.text,
        ocr_confidence=ocr.mean_confidence,
    )
