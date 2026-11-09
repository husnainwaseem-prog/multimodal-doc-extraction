"""LLM / vision-model refinement pass (second stage of the pipeline).

Pipeline: OCR -> rule-based extraction -> LLM/vision verification -> validated JSON.

The rule-based extractor produces a first draft. This stage sends the
OCR text (and, optionally, the page image for vision models) plus the
draft to an LLM and asks for a corrected JSON. The result is never
trusted blindly:

* the reply must parse as JSON and validate against the Invoice model;
* a value that DIFFERS from the rules is only accepted if it is grounded
  in the OCR text (no hallucinated numbers/names);
* agreement between LLM and rules raises a field's confidence;
* any disagreement drops confidence below the review threshold, so a
  human sees it (that is the point of the review flag);
* every decision is written to `result.llm_notes` for audit.

Providers: Anthropic (ANTHROPIC_API_KEY) or any OpenAI-compatible API
(OPENAI_API_KEY). With no key the result is returned unchanged.
For tests, pass `llm_fn(prompt, image_path) -> str` to bypass the network.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from typing import Callable

from .extract import REVIEW_THRESHOLD
from .models import ExtractionResult, Invoice, LineItem

DEFAULT_MODELS = {"anthropic": "claude-sonnet-4-5", "openai": "gpt-4o-mini"}
AGREE_CONF = 0.95      # LLM and rules agree
DISPUTE_CONF = 0.55    # they disagree -> below REVIEW_THRESHOLD -> human review
FILLED_CONF = 0.65     # rules found nothing, LLM filled a grounded value (review)

SCALAR_FIELDS = ["invoice_number", "date", "vendor", "subtotal", "tax", "total"]

PROMPT = """You are verifying structured data extracted from a scanned invoice or receipt.
You get the raw OCR text{vision} and a draft extraction produced by rules.
Fix anything wrong or missing. Use ONLY values that are visible in the document; never invent values.
Return ONLY a JSON object with this schema:
{{"document_type": "invoice|receipt", "invoice_number": str|null, "date": "YYYY-MM-DD"|null,
 "vendor": str|null, "line_items": [{{"description": str, "quantity": number, "unit_price": number, "amount": number}}],
 "subtotal": number|null, "tax": number|null, "total": number|null}}

OCR TEXT:
{ocr}

DRAFT EXTRACTION:
{draft}
"""


def detect_provider() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


def _image_b64(path: str) -> tuple[str, str]:
    media = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return media, base64.b64encode(f.read()).decode()


def call_llm(prompt: str, image_path: str | None, provider: str, model: str | None) -> str:
    """Send the prompt (+ image, for vision) to the provider; return raw text."""
    model = model or DEFAULT_MODELS[provider]
    if provider == "anthropic":
        import anthropic
        content: list[dict] = []
        if image_path:
            media, data = _image_b64(image_path)
            content.append({"type": "image",
                            "source": {"type": "base64", "media_type": media, "data": data}})
        content.append({"type": "text", "text": prompt})
        msg = anthropic.Anthropic().messages.create(
            model=model, max_tokens=2000, temperature=0,
            messages=[{"role": "user", "content": content}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    if provider == "openai":
        from openai import OpenAI
        content = [{"type": "text", "text": prompt}]
        if image_path:
            media, data = _image_b64(image_path)
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{media};base64,{data}"}})
        resp = OpenAI().chat.completions.create(
            model=model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "user", "content": content}])
        return resp.choices[0].message.content
    raise ValueError(f"Unknown provider {provider!r}")


def parse_llm_json(raw: str) -> Invoice:
    """Parse + validate the model reply. Raises ValueError on bad output."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        return Invoice.model_validate(json.loads(raw))
    except Exception as e:  # json error or pydantic ValidationError
        raise ValueError(f"LLM returned invalid JSON: {e}") from e


# --- grounding checks -------------------------------------------------

def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s.lower())


def _grounded(field: str, value, ocr_text: str) -> bool:
    """Is `value` actually visible in the OCR text?"""
    if value is None:
        return False
    if isinstance(value, (int, float)):
        num = f"{float(value):.2f}"
        return num in ocr_text.replace(",", "")
    s = str(value)
    if field == "date":
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})$", s)
        if not m:
            return False
        y, mo, d = m.groups()
        text = ocr_text
        return y in text and (str(int(d)) in text or d in text)
    return _squash(s) in _squash(ocr_text)


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 0.005
    return _squash(str(a)) == _squash(str(b))


def _item_ok(it: LineItem, ocr_text: str) -> bool:
    if it.unit_price is not None and abs(it.quantity * it.unit_price - it.amount) > 0.01:
        return False
    return _grounded("amount", it.amount, ocr_text) and _squash(it.description) in _squash(ocr_text)


def _items_key(items: list[LineItem]) -> list[tuple]:
    return sorted((_squash(i.description), float(i.quantity), round(i.amount, 2)) for i in items)


def merge_llm_result(result: ExtractionResult, llm_inv: Invoice) -> ExtractionResult:
    """Reconcile the LLM's invoice with the rule-based one (see module doc)."""
    inv, conf, notes = result.invoice, result.field_confidence, result.llm_notes
    for f in SCALAR_FIELDS:
        mine, theirs = getattr(inv, f), getattr(llm_inv, f)
        if _same(mine, theirs):
            if mine is not None:
                conf[f] = max(conf.get(f, 0.0), AGREE_CONF)
                notes.append(f"{f}: LLM confirmed")
            continue
        if theirs is None:  # LLM found nothing; keep rules' value
            continue
        if _grounded(f, theirs, result.ocr_text):
            setattr(inv, f, theirs)
            if mine is None:
                conf[f] = FILLED_CONF
                notes.append(f"{f}: filled by LLM ({theirs!r}); please verify")
            else:
                conf[f] = DISPUTE_CONF
                notes.append(f"{f}: LLM changed {mine!r} -> {theirs!r}; please verify")
        else:
            conf[f] = min(conf.get(f, 0.0), DISPUTE_CONF)
            notes.append(f"{f}: LLM suggested {theirs!r} but it is not in the OCR text; kept {mine!r}, flagged")

    # line items: agree -> boost; differ -> take the LLM rows only if every
    # row is arithmetic-consistent and grounded, and flag them for review.
    if _items_key(inv.line_items) == _items_key(llm_inv.line_items):
        for it in inv.line_items:
            it.confidence = max(it.confidence, AGREE_CONF)
        if inv.line_items:
            notes.append("line_items: LLM confirmed")
    elif llm_inv.line_items and all(_item_ok(i, result.ocr_text) for i in llm_inv.line_items):
        for it in llm_inv.line_items:
            it.confidence = DISPUTE_CONF
        inv.line_items = llm_inv.line_items
        notes.append("line_items: LLM version adopted (rows are grounded and add up); please verify")
    elif llm_inv.line_items:
        for it in inv.line_items:
            it.confidence = min(it.confidence, DISPUTE_CONF)
        notes.append("line_items: LLM disagreed but its rows failed grounding/arithmetic; kept rules, flagged")

    optional = {"subtotal", "tax"} if inv.document_type == "receipt" else set()
    review = [f for f in SCALAR_FIELDS
              if conf.get(f, 0.0) < REVIEW_THRESHOLD and not (f in optional and getattr(inv, f) is None)]
    review += [f"line_items[{i}]" for i, it in enumerate(inv.line_items) if it.confidence < REVIEW_THRESHOLD]
    if not inv.line_items:
        review.append("line_items")
    result.needs_review = review
    result.llm_used = True
    return result


def refine_with_llm(result: ExtractionResult, image_path: str | None = None,
                    provider: str | None = None, model: str | None = None,
                    llm_fn: Callable[[str, str | None], str] | None = None) -> ExtractionResult:
    """Run the LLM/vision verification pass on a rule-based `result`.

    image_path: pass the page image to use a vision model (recommended).
    llm_fn: injectable `(prompt, image_path) -> raw reply`, for tests/offline.
    With no API key and no llm_fn the input is returned unchanged.
    """
    if llm_fn is None:
        provider = provider or detect_provider()
        if provider is None:
            result.llm_notes.append("LLM pass skipped: no ANTHROPIC_API_KEY / OPENAI_API_KEY set")
            return result
        llm_fn = lambda prompt, img: call_llm(prompt, img, provider, model)  # noqa: E731

    prompt = PROMPT.format(
        vision=" and the page image" if image_path else "",
        ocr=result.ocr_text,
        draft=result.invoice.model_dump_json(indent=2),
    )
    try:
        llm_inv = parse_llm_json(llm_fn(prompt, image_path))
    except Exception as e:  # network/API/format failure: keep rule-based result
        result.llm_notes.append(f"LLM pass failed ({type(e).__name__}: {e}); rule-based result kept")
        return result
    return merge_llm_result(result, llm_inv)
