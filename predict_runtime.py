"""CPU inference runtime used by the submission notebook.

The recognition path deliberately runs the fine-tuned Korean PP-OCRv5 and
numeric PP-OCRv6 recognizers for every detected text line.  DET is selected
by ``ITDA_DET_MODE`` so the single-detector and full->date cascade variants
share exactly the same REC and date-selection code.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np


# Keep the parser inside the independently cloned submission repository.
_PARSER_ROOT = Path(__file__).resolve().parent
if str(_PARSER_ROOT) not in sys.path:
    sys.path.insert(0, str(_PARSER_ROOT))
from expiry_parser_v4 import (  # noqa: E402
    choose_expiry_date as _choose_expiry_date,
    extract_date_candidates as _extract_date_candidates,
)


REC_FILES = ("inference.json", "inference.pdiparams", "inference.yml")
DET_MODES = {
    "single": ("det_single",),
    "full": ("det_full",),
    "date": ("det_date",),
    "cascade": ("det_full", "det_date"),
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _model_ready(path: Path) -> bool:
    return all((path / name).is_file() for name in REC_FILES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_weights(weights_dir: Path, det_mode: str) -> None:
    if det_mode not in DET_MODES:
        raise ValueError(f"unknown ITDA_DET_MODE: {det_mode!r}")
    required = [weights_dir / "rec_v5", weights_dir / "rec_v6"]
    required += [weights_dir / name for name in DET_MODES[det_mode]]
    missing = [str(path) for path in required if not _model_ready(path)]
    if missing:
        raise FileNotFoundError(
            "offline runtime: model files are missing: " + ", ".join(missing)
            + ". Before disabling internet, run download_weights.sh."
        )

    manifest = weights_dir / "SHA256SUMS"
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 2 or fields[0].startswith("#"):
                continue
            expected, relative = fields
            target = weights_dir / relative
            if target.is_file() and _sha256(target) != expected:
                raise RuntimeError(f"weight checksum mismatch: {relative}")


def load_models(weights_dir: Path, det_mode: str, cpu_threads: int = 4):
    """Load selected DET model(s) and both fixed REC models once."""
    from paddleocr import TextDetection, TextRecognition

    kwargs = dict(device="cpu", enable_mkldnn=False, cpu_threads=cpu_threads)
    recognizers = {
        "v5": TextRecognition(
            model_name="korean_PP-OCRv5_mobile_rec",
            model_dir=str(weights_dir / "rec_v5"),
            **kwargs,
        ),
        "v6": TextRecognition(
            model_name="PP-OCRv6_small_rec",
            model_dir=str(weights_dir / "rec_v6"),
            **kwargs,
        ),
    }
    detectors = {}
    for name in DET_MODES[det_mode]:
        detectors[name] = TextDetection(
            model_name="PP-OCRv6_small_det",
            model_dir=str(weights_dir / name),
            **kwargs,
        )
    return {"detectors": detectors, "recognizers": recognizers}


def detect(image: np.ndarray, detector) -> list[np.ndarray]:
    result = next(iter(detector.predict(image, batch_size=1)))
    boxes = [np.asarray(box, dtype=np.float32) for box in result["dt_polys"]]
    return sorted(boxes, key=lambda box: (float(box[:, 1].mean()), float(box[:, 0].mean())))


def crop_quad(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    points = np.asarray(box, dtype=np.float32).reshape(4, 2)
    width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[3] - points[2])))
    height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    width, height = max(width, 8), max(height, 8)
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(points, target), (width, height))
    if crop.shape[0] > crop.shape[1] * 1.5:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


def padded_crop(image: np.ndarray, box: np.ndarray, pad_x: float = 0.35, pad_y: float = 0.25) -> np.ndarray:
    points = np.asarray(box, dtype=np.float32).reshape(4, 2)
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    height, width = image.shape[:2]
    dx, dy = (x2 - x1) * pad_x, (y2 - y1) * pad_y
    x1, y1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
    x2, y2 = min(width, int(x2 + dx)), min(height, int(y2 + dy))
    return image[y1:y2, x1:x2]


def parse_dates(text: str) -> list[tuple[str, str, str]]:
    """Compatibility wrapper used while comparing the two REC outputs."""
    return list(_extract_date_candidates(text))


def _with_v5_anchor(v6_text: str, v5_text: str) -> str:
    if re.search(r"[가-힣]", v6_text):
        return v6_text
    lead = re.match(r"^\s*([가-힣][가-힣\s:]*)", v5_text)
    tail = re.search(r"([가-힣]+)\s*$", v5_text)
    return "{}{}{}".format(
        (lead.group(1).strip() + " ") if lead else "",
        v6_text,
        tail.group(1) if tail else "",
    )


def _recognize_batch(model, crops: list[np.ndarray]) -> list[tuple[str, float]]:
    if not crops:
        return []
    outputs = model.predict(crops, batch_size=min(16, len(crops)))
    return [(str(item["rec_text"]).strip(), float(item["rec_score"])) for item in outputs]


def recognize_boxes(image: np.ndarray, boxes: list[np.ndarray], models, source: str) -> list[dict]:
    crops = [crop_quad(image, box) for box in boxes]
    v5 = _recognize_batch(models["recognizers"]["v5"], crops)
    v6 = _recognize_batch(models["recognizers"]["v6"], crops)
    records = []
    for index, box in enumerate(boxes):
        text5, score5 = v5[index]
        text6, score6 = v6[index]
        dates5, dates6 = set(parse_dates(text5)), set(parse_dates(text6))
        use_v6 = bool(dates6) and (not dates5 or (dates6 != dates5 and score6 > score5))
        text, score, variant = (
            (_with_v5_anchor(text6, text5), score6, "v6") if use_v6
            else (text5, score5, "v5")
        )
        records.append({
            "text": text, "conf": score, "variant": variant, "source": source,
            "v5_text": text5, "v6_text": text6, "v5_conf": score5, "v6_conf": score6,
            "box": np.asarray(box).tolist(), "joined": False,
        })
    return records


def _joined_records(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        box = np.asarray(record["box"], dtype=np.float32)
        cy, height = float(box[:, 1].mean()), float(box[:, 1].max() - box[:, 1].min())
        row = next((item for item in rows if abs(cy - item["cy"]) <= 0.8 * max(height, item["height"])), None)
        if row is None:
            rows.append({"records": [record], "cy": cy, "height": max(height, 1.0)})
        else:
            row["records"].append(record)
            row["cy"] = float(np.mean([np.asarray(item["box"])[:, 1].mean() for item in row["records"]]))
            row["height"] = max(row["height"], height)
    joined = []
    for row in rows:
        items = sorted(row["records"], key=lambda item: float(np.asarray(item["box"])[:, 0].min()))
        if len(items) < 2:
            continue
        joined.append({
            "text": " ".join(item["text"] for item in items),
            "conf": float(np.mean([item["conf"] for item in items])),
            "variant": "+".join(sorted({item["variant"] for item in items})),
            "source": items[0]["source"], "joined": True,
            "box": np.concatenate([np.asarray(item["box"]) for item in items]).tolist(),
        })
    return joined


def choose(records: list[dict]):
    all_records = list(records) + _joined_records(records)
    return _choose_expiry_date(
        [record["text"] for record in all_records],
        rec_confs=[float(record.get("conf", 1.0)) for record in all_records],
        boxes=[record.get("box") for record in all_records],
    )


def predict_image(image: np.ndarray, models, det_mode: str):
    if det_mode == "cascade":
        full_boxes = detect(image, models["detectors"]["det_full"])
        records = recognize_boxes(image, full_boxes, models, "full")
        for full_box in full_boxes[:4]:
            full_crop = crop_quad(image, full_box)
            date_boxes = detect(full_crop, models["detectors"]["det_date"])
            records.extend(recognize_boxes(full_crop, date_boxes, models, "date_in_full"))
        boxes = full_boxes
    else:
        detector_name = DET_MODES[det_mode][0]
        boxes = detect(image, models["detectors"][detector_name])
        records = recognize_boxes(image, boxes, models, detector_name)

    winner = choose(records)
    if winner is None and boxes:
        retry = []
        for box in boxes[:2]:
            crop = padded_crop(image, box)
            if crop.size:
                retry.extend(recognize_boxes(crop, [np.array([[0, 0], [crop.shape[1] - 1, 0], [crop.shape[1] - 1, crop.shape[0] - 1], [0, crop.shape[0] - 1]], dtype=np.float32)], models, "retry"))
        winner = choose(records + retry)
    return winner["date"] if winner else ("NONE", "NONE", "NONE")


def list_images(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def final_date(item: tuple[str, str, str]) -> str:
    return "NONE" if all(value == "NONE" for value in item) else "-".join(item)
