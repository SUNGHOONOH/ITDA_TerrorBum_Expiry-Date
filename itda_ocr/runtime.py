"""Offline CPU inference runtime used by the submission notebook.

The runtime uses one fine-tuned detector, both fine-tuned recognizers, and the
small preprocessing/rescue path validated in the fourth-round test. All model
directories are local; this module never downloads weights.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import cv2
import numpy as np


from .parser import (
    choose_expiry_date as _choose_expiry_date,
    extract_date_candidates as _extract_date_candidates,
)


REC_FILES = ("inference.json", "inference.pdiparams", "inference.yml")
UVDOC_FILES = ("model.safetensors", "config.json", "preprocessor_config.json", "inference.yml")
MODEL_FILES = {
    "det_single": REC_FILES,
    "rec_v5": REC_FILES,
    "rec_v6": REC_FILES,
    "textline_ori": REC_FILES,
    "uvdoc": UVDOC_FILES,
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _model_ready(path: Path, required_files=REC_FILES) -> bool:
    return all((path / name).is_file() for name in required_files)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_weights(weights_dir: Path) -> None:
    missing = [
        str(weights_dir / name)
        for name, files in MODEL_FILES.items()
        if not _model_ready(weights_dir / name, files)
    ]
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


def load_models(weights_dir: Path, cpu_threads: int = 4):
    """Load all local models once; the expensive UVDoc model stays lazy."""
    from paddleocr import TextDetection, TextLineOrientationClassification, TextRecognition

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
    detector = TextDetection(
        model_name="PP-OCRv6_small_det",
        model_dir=str(weights_dir / "det_single"),
        limit_side_len=1280,
        limit_type="max",
        box_thresh=0.30,
        **kwargs,
    )
    orientation = TextLineOrientationClassification(
        model_name="PP-LCNet_x0_25_textline_ori",
        model_dir=str(weights_dir / "textline_ori"),
        **kwargs,
    )
    return {
        "detector": detector,
        "orientation": orientation,
        "recognizers": recognizers,
        "uvdoc": None,
        "uvdoc_error": "",
        "uvdoc_dir": weights_dir / "uvdoc",
    }


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


CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


def _upscale(image: np.ndarray, target_height: int = 96) -> np.ndarray:
    scale = min(3.0, target_height / max(1, image.shape[0]))
    if scale <= 1:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _dark_text(gray: np.ndarray) -> np.ndarray:
    bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0
    return 255 - gray if bright.mean() < 0.5 else gray


def _view_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[..., 0] = CLAHE.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _view_dot_close(image: np.ndarray) -> np.ndarray:
    gray = CLAHE.apply(_dark_text(_to_gray(image)))
    inverse = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    size = 2 if gray.shape[0] < 120 else 3
    joined = cv2.morphologyEx(
        inverse,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
    )
    return 255 - joined


def _rescue_crop(image: np.ndarray, box: np.ndarray, pad_x: float = 1.0, pad_y: float = 0.25) -> np.ndarray:
    points = np.asarray(box, dtype=np.int32)
    x, y, width, height = cv2.boundingRect(points)
    px, py = int(round(height * pad_x)), int(round(height * pad_y))
    return image[max(0, y - py):y + height + py, max(0, x - px):x + width + px].copy()


def _make_views(image: np.ndarray, box: np.ndarray) -> list[tuple[str, np.ndarray]]:
    base = _upscale(_rescue_crop(image, box))
    return [("pad", base), ("clahe", _view_clahe(base)), ("dot_close", _to_bgr(_view_dot_close(base)))]


def _orient_crops(crops: list[np.ndarray], orientation) -> list[np.ndarray]:
    if not crops:
        return []
    outputs = orientation.predict(crops, batch_size=min(16, len(crops)))
    oriented = []
    for crop, item in zip(crops, outputs):
        labels = item.get("label_names", [])
        label = str(labels[0]) if labels else "0_degree"
        oriented.append(cv2.rotate(crop, cv2.ROTATE_180) if label.startswith("180") else crop)
    return oriented


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


def recognize_boxes(
    image: np.ndarray,
    boxes: list[np.ndarray],
    models,
    source: str,
    box_offset: int = 0,
) -> list[dict]:
    crops = _orient_crops([crop_quad(image, box) for box in boxes], models["orientation"])
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
            "box": np.asarray(box).tolist(), "box_index": box_offset + index, "joined": False,
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


def _box_iou(first: np.ndarray, second: np.ndarray) -> float:
    ax, ay, aw, ah = cv2.boundingRect(np.asarray(first, dtype=np.int32))
    bx, by, bw, bh = cv2.boundingRect(np.asarray(second, dtype=np.int32))
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by)
    )
    return intersection / max(1, aw * ah + bw * bh - intersection)


def _choose_confidence(records: list[dict], winner: dict | None) -> float:
    if not winner:
        return 0.0
    matching = [record for record in records if record.get("text") == winner.get("text")]
    if not matching:
        matching = [record for record in records if record.get("text") in str(winner.get("text", ""))]
    return max((float(record.get("conf", 0.0)) for record in matching), default=0.0)


def _get_uvdoc(models):
    if models["uvdoc"] is not None or models["uvdoc_error"]:
        return models["uvdoc"]
    try:
        from paddleocr import TextImageUnwarping

        models["uvdoc"] = TextImageUnwarping(
            model_name="UVDoc",
            model_dir=str(models["uvdoc_dir"]),
            device="cpu",
            enable_mkldnn=False,
            cpu_threads=4,
        )
    except Exception as error:  # pragma: no cover - only used for an invalid local export
        models["uvdoc_error"] = f"{type(error).__name__}: {error}"
    return models["uvdoc"]


def _rescue(
    image: np.ndarray,
    models,
    boxes: list[np.ndarray],
    records: list[dict],
    winner: dict | None,
    confidence: float,
) -> tuple[dict | None, float]:
    """Retry low-confidence cases with contrast, padding, and local UVDoc."""
    if winner is None:
        extra = [
            box
            for box in detect(_view_clahe(image), models["detector"])
            if all(_box_iou(box, old) < 0.5 for old in boxes)
        ]
        if extra:
            records.extend(recognize_boxes(image, extra, models, "det_clahe", len(boxes)))
            boxes.extend(extra)
            winner = choose(records)
            confidence = _choose_confidence(records, winner)

    base = [
        index
        for index, record in enumerate(records)
        if record.get("variant") in {"raw", "v6", "det_clahe"}
        and record.get("box_index", -1) < len(boxes)
    ]
    if winner:
        base.sort(key=lambda index: records[index].get("text") != winner.get("text"))
    targets = base[:2]

    if (winner is None or confidence < 0.50) and targets:
        views = [
            (index, name, view)
            for index in targets
            for name, view in _make_views(image, boxes[records[index]["box_index"]])
        ]
        texts = _recognize_batch(models["recognizers"]["v5"], [view for _, _, view in views])
        for (index, name, _), (text, score) in zip(views, texts):
            original = records[index]
            records.append({
                "text": text,
                "conf": score,
                "variant": name,
                "source": "rescue",
                "v5_text": text,
                "v6_text": "",
                "box": original["box"],
                "box_index": original["box_index"],
                "joined": False,
            })
        winner = choose(records)
        confidence = _choose_confidence(records, winner)

    if (winner is None or confidence < 0.50) and targets:
        uvdoc = _get_uvdoc(models)
        if uvdoc is not None:
            first = records[targets[0]]
            crop = _rescue_crop(image, boxes[first["box_index"]])
            try:
                output = next(iter(uvdoc.predict(crop, batch_size=1)))
                flat = np.asarray(output["doctr_img"])
                if flat.ndim == 3 and flat.size:
                    if flat.dtype != np.uint8:
                        flat = np.clip(flat * (255.0 if flat.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
                    flat = cv2.cvtColor(flat, cv2.COLOR_RGB2BGR)
                    variants = [("uvdoc", flat), ("uvdoc_clahe", _view_clahe(flat))]
                    texts = _recognize_batch(models["recognizers"]["v5"], [view for _, view in variants])
                    for (name, _), (text, score) in zip(variants, texts):
                        records.append({
                            "text": text,
                            "conf": score,
                            "variant": name,
                            "source": "uvdoc",
                            "v5_text": text,
                            "v6_text": "",
                            "box": first["box"],
                            "box_index": first["box_index"],
                            "joined": False,
                        })
                    winner = choose(records)
                    confidence = _choose_confidence(records, winner)
            except Exception as error:  # pragma: no cover - depends on malformed input crop
                models["uvdoc_error"] = f"{type(error).__name__}: {error}"
    return winner, confidence


def predict_image(image: np.ndarray, models):
    boxes = detect(image, models["detector"])
    records = recognize_boxes(image, boxes, models, "det_single")

    winner = choose(records)
    confidence = _choose_confidence(records, winner)
    if winner is None or confidence < 0.50:
        winner, confidence = _rescue(image, models, boxes, records, winner, confidence)
    return winner["date"] if winner else ("NONE", "NONE", "NONE")


def list_images(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def final_date(item: tuple[str, str, str]) -> str:
    return "NONE" if all(value == "NONE" for value in item) else "-".join(item)
