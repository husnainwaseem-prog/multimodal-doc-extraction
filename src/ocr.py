"""OCR backends.

Two engines, one interface:
- "tesseract" — the classic OCR engine (needs the `tesseract` binary).
- "rapidocr"  — ONNX-based, pip-installable, no system binary needed.

Both return the full text plus per-word confidence scores, which the
extractor uses to score each field. The default engine is RapidOCR
because it needs no system install; pass --engine tesseract to use
Tesseract instead.

Line reconstruction: OCR engines return text in block order, which
splits table rows across lines (each column becomes its own line).
We rebuild visual lines from the word bounding boxes — words whose
y-centres are close belong to one line, sorted left to right — so
"Desk Lamp LED  1  32.00  32.00" comes back as a single row.
"""

from __future__ import annotations

import shutil


class OCRResult:
    def __init__(self, text: str, words: list[tuple[str, float]], lines: list[str] | None = None):
        self.text = text
        self.words = words  # (word, confidence 0..1)
        self.lines = lines if lines is not None else text.splitlines()

    @property
    def mean_confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(c for _, c in self.words) / len(self.words)


def _rebuild_lines(boxes: list[tuple[float, float, str, float]]) -> list[str]:
    """Group (x, y_centre, text, conf) into visual lines.

    Words within 12px of y-centre join one line; lines sorted top to
    bottom, words left to right.
    """
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    lines: list[list[tuple[float, float, str, float]]] = []
    for b in boxes:
        placed = False
        for line in lines:
            y0 = sum(x[1] for x in line) / len(line)
            if abs(b[1] - y0) <= 12:
                line.append(b)
                placed = True
                break
        if not placed:
            lines.append([b])
    out = []
    for line in lines:
        line.sort(key=lambda b: b[0])
        out.append(" ".join(b[2] for b in line))
    return out


def ocr_tesseract(image_path: str) -> OCRResult:
    """OCR with Tesseract via pytesseract."""
    import pytesseract
    from PIL import Image

    img = Image.open(image_path)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    words: list[tuple[str, float]] = []
    boxes: list[tuple[float, float, str, float]] = []
    for word, conf, left, top, w, h in zip(
        data["text"], data["conf"], data["left"], data["top"], data["width"], data["height"]
    ):
        word = word.strip()
        if not word:
            continue
        try:
            c = float(conf) / 100.0
        except (TypeError, ValueError):
            c = 0.0
        c = max(0.0, min(1.0, c))
        words.append((word, c))
        boxes.append((float(left), float(top) + float(h) / 2, word, c))
    lines = _rebuild_lines(boxes)
    text = pytesseract.image_to_string(img)
    return OCRResult(text="\n".join(lines), words=words, lines=lines)


def ocr_rapidocr(image_path: str) -> OCRResult:
    """OCR with RapidOCR (ONNX)."""
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    result, _ = engine(image_path)
    words: list[tuple[str, float]] = []
    boxes: list[tuple[float, float, str, float]] = []
    if result:
        for box, text, conf in result:
            text = text.strip()
            if not text:
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x_min, y_centre = min(xs), (min(ys) + max(ys)) / 2
            for w in text.split():
                words.append((w, float(conf)))
                boxes.append((x_min, y_centre, w, float(conf)))
    lines = _rebuild_lines(boxes)
    return OCRResult(text="\n".join(lines), words=words, lines=lines)


def ocr_image(image_path: str, engine: str = "rapidocr") -> OCRResult:
    """Run OCR on one image file. engine: 'rapidocr' or 'tesseract'."""
    if engine == "tesseract":
        if shutil.which("tesseract") is None:
            raise RuntimeError(
                "Tesseract binary not found. Install it "
                "(https://github.com/tesseract-ocr/tesseract) "
                "or use --engine rapidocr."
            )
        return ocr_tesseract(image_path)
    if engine == "rapidocr":
        return ocr_rapidocr(image_path)
    raise ValueError(f"Unknown OCR engine: {engine!r} (use 'rapidocr' or 'tesseract')")


def available_engines() -> list[str]:
    engines = []
    try:
        import rapidocr_onnxruntime  # noqa: F401
        engines.append("rapidocr")
    except ImportError:
        pass
    if shutil.which("tesseract") is not None:
        engines.append("tesseract")
    return engines
