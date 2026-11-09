"""Synthetic invoice / receipt generator.

Renders clean, OCR-friendly document images with PIL and writes the
exact ground truth for each one. 30 documents by default: 20 invoices
+ 10 receipts, seeded so the set is reproducible.

Why synthetic: real scanned invoices cannot be redistributed, and
synthetic ground truth is exact — so the field-level accuracy numbers
in the report are measured against known truth, not hand labels.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

VENDORS = [
    "Acme Office Supplies", "BrightTech Solutions", "Global Hardware Ltd",
    "Northwind Traders", "Pinnacle Electronics", "Summit Stationery Co",
    "Vertex Computing", "BluePeak Distributors", "Metro Office Mart",
    "Quantum Parts Inc", "Evergreen Paper Co", "SwiftLogix Systems",
]

ITEM_POOL = [
    ("Wireless Mouse", 25.00), ("USB-C Cable 1m", 12.50), ("A4 Paper Ream", 8.99),
    ("Ballpoint Pen (Box)", 6.50), ("Mechanical Keyboard", 59.99),
    ("HDMI Cable 2m", 15.00), ("Notebook A5", 4.25), ("Desk Lamp LED", 32.00),
    ("Monitor Stand", 45.00), ("Printer Ink Cartridge", 39.95),
    ("Stapler Heavy Duty", 18.75), ("External SSD 500GB", 79.99),
    ("Webcam HD 1080p", 49.50), ("Laptop Sleeve 15in", 22.00),
    ("Whiteboard Markers", 9.99), ("Filing Cabinet", 129.00),
]

DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d %b %Y"]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def render_invoice(vendor: str, inv_no: str, date_str: str, items: list[dict],
                   tax_rate: float, path: Path) -> dict:
    """Render one invoice image; return its ground truth dict."""
    W, H = 900, 700
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    y = 40
    d.text((50, y), vendor, font=_font(28, bold=True), fill="black")
    y += 60
    d.text((50, y), "INVOICE", font=_font(22, bold=True), fill="black")
    y += 50
    d.text((50, y), f"Invoice No: {inv_no}", font=_font(18), fill="black")
    y += 35
    d.text((50, y), f"Date: {date_str}", font=_font(18), fill="black")
    y += 60
    # table header
    d.text((50, y), "Description", font=_font(18, bold=True), fill="black")
    d.text((450, y), "Qty", font=_font(18, bold=True), fill="black")
    d.text((560, y), "Unit Price", font=_font(18, bold=True), fill="black")
    d.text((730, y), "Amount", font=_font(18, bold=True), fill="black")
    y += 35
    d.line([(50, y), (850, y)], fill="black", width=2)
    y += 15
    subtotal = 0.0
    for it in items:
        amt = round(it["qty"] * it["unit_price"], 2)
        subtotal += amt
        d.text((50, y), it["description"], font=_font(18), fill="black")
        d.text((450, y), str(it["qty"]), font=_font(18), fill="black")
        d.text((560, y), f"{it['unit_price']:.2f}", font=_font(18), fill="black")
        d.text((730, y), f"{amt:.2f}", font=_font(18), fill="black")
        y += 35
    y += 10
    d.line([(50, y), (850, y)], fill="black", width=2)
    y += 20
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)
    d.text((560, y), "Subtotal", font=_font(18), fill="black")
    d.text((730, y), f"{subtotal:.2f}", font=_font(18), fill="black")
    y += 35
    d.text((560, y), "Tax", font=_font(18), fill="black")
    d.text((730, y), f"{tax:.2f}", font=_font(18), fill="black")
    y += 35
    d.text((560, y), "Total", font=_font(18, bold=True), fill="black")
    d.text((730, y), f"{total:.2f}", font=_font(18, bold=True), fill="black")
    y += 60
    d.text((50, y), "Thank you for your business.", font=_font(16), fill="black")
    img.save(path)
    return {
        "file": path.name,
        "document_type": "invoice",
        "invoice_number": inv_no,
        "date": date_str,
        "vendor": vendor,
        "line_items": [
            {"description": it["description"], "quantity": it["qty"],
             "unit_price": it["unit_price"], "amount": round(it["qty"] * it["unit_price"], 2)}
            for it in items
        ],
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "currency": "USD",
    }


def render_receipt(vendor: str, inv_no: str, date_str: str, items: list[dict],
                   path: Path) -> dict:
    """Render one receipt image; return its ground truth dict."""
    W, H = 700, 600
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    y = 40
    d.text((50, y), vendor, font=_font(26, bold=True), fill="black")
    y += 55
    d.text((50, y), "RECEIPT", font=_font(20, bold=True), fill="black")
    y += 45
    d.text((50, y), f"Receipt No: {inv_no}", font=_font(18), fill="black")
    y += 35
    d.text((50, y), f"Date: {date_str}", font=_font(18), fill="black")
    y += 50
    d.line([(50, y), (650, y)], fill="black", width=2)
    y += 15
    total = 0.0
    for it in items:
        amt = round(it["qty"] * it["unit_price"], 2)
        total += amt
        d.text((50, y), it["description"], font=_font(18), fill="black")
        d.text((450, y), f"{it['qty']} x {it['unit_price']:.2f}", font=_font(18), fill="black")
        d.text((580, y), f"{amt:.2f}", font=_font(18), fill="black")
        y += 35
    y += 10
    d.line([(50, y), (650, y)], fill="black", width=2)
    y += 20
    d.text((450, y), "Total", font=_font(18, bold=True), fill="black")
    d.text((580, y), f"{total:.2f}", font=_font(18, bold=True), fill="black")
    y += 60
    d.text((50, y), "Thank you for shopping with us.", font=_font(16), fill="black")
    img.save(path)
    return {
        "file": path.name,
        "document_type": "receipt",
        "invoice_number": inv_no,
        "date": date_str,
        "vendor": vendor,
        "line_items": [
            {"description": it["description"], "quantity": it["qty"],
             "unit_price": it["unit_price"], "amount": round(it["qty"] * it["unit_price"], 2)}
            for it in items
        ],
        "subtotal": None,
        "tax": None,
        "total": total,
        "currency": "USD",
    }


def generate_dataset(n_invoices: int = 20, n_receipts: int = 10, seed: int = 42,
                     out_dir: str = "data/samples") -> list[dict]:
    """Generate the full sample set and write ground_truth.jsonl."""
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    truth: list[dict] = []

    for i in range(n_invoices):
        vendor = rng.choice(VENDORS)
        inv_no = f"INV-{20260000 + i * 7 + rng.randint(1, 6)}"
        fmt = rng.choice(DATE_FORMATS)
        from datetime import date, timedelta
        d = date(2026, 1, 1) + timedelta(days=rng.randint(0, 300))
        date_str = d.strftime(fmt)
        n_items = rng.randint(2, 5)
        items = []
        for desc, price in rng.sample(ITEM_POOL, n_items):
            items.append({"description": desc, "qty": rng.randint(1, 4), "unit_price": price})
        gt = render_invoice(vendor, inv_no, date_str, items, 0.08, out / f"invoice_{i+1:02d}.png")
        truth.append(gt)

    for i in range(n_receipts):
        vendor = rng.choice(VENDORS)
        inv_no = f"RCP-{50000 + i * 13 + rng.randint(1, 9)}"
        fmt = rng.choice(DATE_FORMATS)
        from datetime import date, timedelta
        d = date(2026, 1, 1) + timedelta(days=rng.randint(0, 300))
        date_str = d.strftime(fmt)
        n_items = rng.randint(2, 4)
        items = []
        for desc, price in rng.sample(ITEM_POOL, n_items):
            items.append({"description": desc, "qty": rng.randint(1, 3), "unit_price": price})
        gt = render_receipt(vendor, inv_no, date_str, items, out / f"receipt_{i+1:02d}.png")
        truth.append(gt)

    gt_path = Path(out_dir).parent / "ground_truth.jsonl"
    with open(gt_path, "w") as f:
        for row in truth:
            f.write(json.dumps(row) + "\n")
    return truth
