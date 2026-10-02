"""필드 검출 단계: YOLO11n(수평 박스, expiry_field/lot_field) + 단계형(cascade) 결합.

YOLO → 영역 안에서 범용 DB 검출기로 글자 줄 4점 다시 찾기 → runtime의 방향 보정 + v5/v6 REC
→ 글자 형식으로 역할 재확인 → 날짜 파서로 사용기한 선택, LOT 값 추출.

제출(predict.ipynb): predict_cascade() — 1단계 runtime.predict_image 가 NONE일 때만 이 단계를 돌린다.
데모(demo/phone_server.py): predict_fields() — 항상 YOLO를 돌려 사용기한 + 제조번호를 낸다.
추론은 onnxruntime만 쓴다 (ultralytics는 GUI용 opencv를 끌고 와 헤드리스 환경에서 import가 죽는다).
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from . import runtime as rt

CLASSES = ("expiry", "lot")
YOLO_IMGSZ = 1280  # 학습 imgsz와 반드시 같게 (Ultralytics: 학습과 같은 크기에서 추론할 때 최선)

LOT_CUE = re.compile(r"제조번호|제조\s*번호|LOT|Lot|lot|BATCH|Batch|배치|로트")
SERIAL_CUE = re.compile(r"일련번호|S/N|SN")
DATE_SHAPE = re.compile(r"(?:19|20)?\d{2}\s*[./\-년]\s*\d{1,2}(?:\s*[./\-월]\s*\d{1,2})?")


# ---------------------------------------------------------------- YOLO (onnxruntime)
def letterbox(image: np.ndarray, size: int = YOLO_IMGSZ):
    """Ultralytics와 같은 가운데 정렬 letterbox (패딩 114)."""
    h, w = image.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, r, left, top


def decode_yolo(output, r: float, left: int, top: int, image_shape, conf: float = 0.25, iou: float = 0.5) -> list[dict]:
    """YOLO11 export (1, 4+nc, N) 와 YOLO26 end2end export (1, 300, 6) 를 모두 처리."""
    out = np.asarray(output, dtype=np.float32)
    out = out[0] if out.ndim == 3 else out
    if out.shape[-1] == 6 and out.shape[0] != 4 + len(CLASSES):   # end2end: x1,y1,x2,y2,score,cls
        keep = out[:, 4] >= conf
        boxes, scores, cls = out[keep, :4], out[keep, 4], out[keep, 5].astype(int)
    else:                                                           # (4+nc, N): cx,cy,w,h,score...
        out = out.T if out.shape[0] == 4 + len(CLASSES) else out
        cls_scores = out[:, 4:]
        cls, scores = cls_scores.argmax(1), cls_scores.max(1)
        keep = scores >= conf
        xywh, cls, scores = out[keep, :4], cls[keep], scores[keep]
        boxes = np.column_stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                                 xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2])
        kept: list[int] = []
        for c in np.unique(cls):
            idx = np.where(cls == c)[0]
            rects = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes[idx]]
            chosen = cv2.dnn.NMSBoxes(rects, scores[idx].tolist(), conf, iou)
            kept.extend(idx[np.asarray(chosen, dtype=int).reshape(-1)].tolist())
        kept.sort()
        boxes, scores, cls = boxes[kept], scores[kept], cls[kept]
    h, w = image_shape[:2]
    boxes = (boxes - np.array([left, top, left, top], dtype=np.float32)) / r
    boxes = np.clip(boxes, 0, [w - 1, h - 1, w - 1, h - 1])
    return [{"cls": CLASSES[int(c)] if 0 <= int(c) < len(CLASSES) else str(int(c)),
             "score": float(s), "xyxy": [float(v) for v in b]}
            for b, s, c in zip(boxes, scores, cls) if b[2] - b[0] >= 2 and b[3] - b[1] >= 2]


class YoloFieldDetector:
    def __init__(self, onnx_path: Path, cpu_threads: int = 4, imgsz: int = YOLO_IMGSZ):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = cpu_threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        side = self.session.get_inputs()[0].shape[2]
        self.imgsz = side if isinstance(side, int) else imgsz  # 고정 크기로 export한 ONNX면 그 크기를 따른다

    def __call__(self, image: np.ndarray, conf: float = 0.25) -> list[dict]:
        canvas, r, left, top = letterbox(image, self.imgsz)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        output = self.session.run(None, {self.input_name: blob})[0]
        return decode_yolo(output, r, left, top, image.shape, conf=conf)


# ---------------------------------------------------------------- 영역 안 4점 다듬기
def order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype=np.float32)


def refine_quad(image: np.ndarray, xyxy, refiner, pad_x: float = 0.10, pad_y: float = 0.30):
    """YOLO 박스를 조금 넓혀 자르고 범용 DB로 글자 줄 4점을 찾는다. 못 찾으면 YOLO 박스 그대로."""
    x1, y1, x2, y2 = xyxy
    rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    if refiner is None:
        return rect, "yolo_box"
    H, W = image.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    X1, Y1 = max(0, int(x1 - bw * pad_x)), max(0, int(y1 - bh * pad_y))
    X2, Y2 = min(W, int(x2 + bw * pad_x)), min(H, int(y2 + bh * pad_y))
    if X2 - X1 < 8 or Y2 - Y1 < 8:
        return rect, "yolo_box"
    inside = []
    for poly in rt.detect(image[Y1:Y2, X1:X2], refiner):
        poly = poly + np.array([X1, Y1], dtype=np.float32)
        cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            inside.append(poly)
    if not inside:
        return rect, "yolo_box"
    if len(inside) == 1:
        return order_quad(inside[0]), "db_refined"          # 원근까지 살아 있는 4점
    merged = cv2.boxPoints(cv2.minAreaRect(np.concatenate(inside).astype(np.float32)))
    return order_quad(merged), "db_merged"                  # 단어 여러 개 → 회전 사각형


# ---------------------------------------------------------------- 형식 기반 역할 확인 (파서와 같은 규칙)
def has_date(text: str) -> bool:
    return bool(rt.parse_dates(text))


def lot_value(text: str) -> str | None:
    """단서·구분기호를 지우고 LOT꼴(영숫자 3~15자, 숫자 포함, 날짜·일련번호·바코드 아님) 첫 값을 돌려준다."""
    if SERIAL_CUE.search(text) and not LOT_CUE.search(text):
        return None
    cleaned = DATE_SHAPE.sub(" ", LOT_CUE.sub(" ", text))    # 날짜를 먼저 지워야 "2024"가 LOT로 안 잡힘
    cleaned = re.sub(r"[:：/|]", " ", cleaned)
    for token in re.findall(r"[A-Za-z0-9\-]{3,15}", cleaned):
        if not re.search(r"\d", token):
            continue
        if re.fullmatch(r"\d{12,}", token) or re.fullmatch(r"880\d{10}", token):
            continue
        if DATE_SHAPE.fullmatch(token) or has_date(token):
            continue
        return token
    return None


def verify_role(yolo_cls: str, text: str) -> str:
    """YOLO 클래스를 글자 형식으로 재확인. 날짜가 없는데 LOT꼴만 있으면 lot, 그 반대면 expiry."""
    dated, lot = has_date(text), lot_value(text)
    if yolo_cls == "expiry" and not dated and lot:
        return "lot"
    if yolo_cls == "lot" and dated and not lot:
        return "expiry"
    return yolo_cls


# ---------------------------------------------------------------- 전체 파이프라인
def load_demo_models(weights_dir: Path, yolo_onnx: Path, refiner_dir: Path | None = None, cpu_threads: int = 4):
    """weights_dir: 제출 runtime과 같은 구조 (det_single, rec_v5, rec_v6, textline_ori, uvdoc).
    새 REC를 쓰려면 rec_v5/rec_v6만 교체한 복사본 폴더를 넘긴다."""
    from paddleocr import TextDetection

    models = rt.load_models(weights_dir, cpu_threads=cpu_threads)
    models["yolo"] = YoloFieldDetector(yolo_onnx, cpu_threads=cpu_threads)
    # 제출용 det_single은 "소비기한 줄만" 학습된 검출기라 LOT 줄을 못 찾을 수 있어 범용 모델을 쓴다.
    kwargs = dict(device="cpu", enable_mkldnn=False, cpu_threads=cpu_threads)
    try:
        models["refiner"] = TextDetection(model_name="PP-OCRv6_small_det",
                                          model_dir=str(refiner_dir) if refiner_dir else None, **kwargs)
        models["refiner_name"] = "PP-OCRv6_small_det (범용)"
    except Exception as error:  # 오프라인에서 범용 가중치가 없을 때
        models["refiner"] = models["detector"]
        models["refiner_name"] = f"det_single 폴백 ({type(error).__name__})"
    return models


def predict_fields(image: np.ndarray, models, conf: float = 0.25) -> dict:
    fields = models["yolo"](image, conf=conf)
    quads, meta = [], []
    for field in fields:
        quad, how = refine_quad(image, field["xyxy"], models.get("refiner"))
        quads.append(quad)
        meta.append({**field, "refine": how})
    records = rt.recognize_boxes(image, quads, models, "yolo") if quads else []

    expiry_records, lot_candidates, reclassified = [], [], 0
    for record, info in zip(records, meta):
        texts = [record["text"], record.get("v6_text", ""), record.get("v5_text", "")]
        role = verify_role(info["cls"], record["text"])
        reclassified += role != info["cls"]
        record.update(yolo_cls=info["cls"], role=role, yolo_score=info["score"], refine=info["refine"])
        if role == "expiry":
            expiry_records.append(record)
        elif role == "lot":
            value = next((v for v in (lot_value(t) for t in texts if t) if v), None)
            conf_lot = max(record.get("v6_conf", 0.0), record.get("v5_conf", 0.0))
            if value:
                lot_candidates.append((conf_lot * info["score"], value, record))

    winner = rt.choose(expiry_records) if expiry_records else None
    expiry_conf = rt._choose_confidence(expiry_records, winner) if winner else 0.0
    used_fallback = False
    if winner is None:  # YOLO가 사용기한을 못 찾으면 기존 제출 파이프라인으로 날짜만 다시 시도
        ymd = rt.predict_image(image, models)
        used_fallback = ymd != ("NONE", "NONE", "NONE")
    else:
        ymd = winner["date"]
    lot = max(lot_candidates, key=lambda item: item[0]) if lot_candidates else None

    review = []
    if ymd == ("NONE", "NONE", "NONE"):
        review.append("사용기한 없음")
    elif "NONE" in ymd:
        review.append("부분 날짜")
    if winner is not None and expiry_conf < 0.5:
        review.append("사용기한 신뢰도 낮음")
    if used_fallback:
        review.append("YOLO 미검출 → 기존 파이프라인")
    if any(m["cls"] == "lot" for m in meta) and lot is None:
        review.append("LOT 박스는 있으나 값 판독 실패")
    if reclassified:
        review.append(f"역할 재분류 {reclassified}건")

    return {
        "expiry": ymd,
        "expiry_final": rt.final_date(ymd),
        "expiry_conf": round(expiry_conf, 3),
        "lot": lot[1] if lot else "",
        "lot_conf": round(lot[0], 3) if lot else 0.0,
        "needs_review": "; ".join(review),
        "fields": [{k: v for k, v in r.items() if k in ("text", "conf", "role", "yolo_cls", "yolo_score", "refine", "box")}
                   for r in records],
    }


# ---------------------------------------------------------------- 제출용 단계형 파이프라인
NONE_DATE = ("NONE", "NONE", "NONE")
FIELD_MODEL_FILES = {
    "yolo_field": ("best.onnx",),
    "refiner_det": rt.REC_FILES,
    "rec_v5_med": rt.REC_FILES,
    "rec_v6_med": rt.REC_FILES,
}


def ensure_field_weights(weights_dir: Path) -> None:
    missing = [str(weights_dir / name / file) for name, files in FIELD_MODEL_FILES.items()
               for file in files if not (weights_dir / name / file).is_file()]
    if missing:
        raise FileNotFoundError("offline runtime: field-stage model files are missing: " + ", ".join(missing)
                                + ". Before disabling internet, run download_weights.sh.")


def load_cascade_models(weights_dir: Path, cpu_threads: int = 4):
    """1단계(4차 DET + 4차 REC)와 2단계(YOLO + 범용 DET 보정 + 의약품까지 학습한 5차 REC)를 한 번에 불러온다."""
    from paddleocr import TextDetection, TextRecognition

    ensure_field_weights(weights_dir)
    stage1 = rt.load_models(weights_dir, cpu_threads=cpu_threads)
    kwargs = dict(device="cpu", enable_mkldnn=False, cpu_threads=cpu_threads)
    stage2 = dict(stage1)   # 방향 분류기는 같이 쓴다
    stage2["recognizers"] = {
        "v5": TextRecognition(model_name="korean_PP-OCRv5_mobile_rec", model_dir=str(weights_dir / "rec_v5_med"), **kwargs),
        "v6": TextRecognition(model_name="PP-OCRv6_small_rec", model_dir=str(weights_dir / "rec_v6_med"), **kwargs),
    }
    stage2["yolo"] = YoloFieldDetector(weights_dir / "yolo_field" / "best.onnx", cpu_threads=cpu_threads)
    stage2["refiner"] = TextDetection(model_name="PP-OCRv6_small_det", model_dir=str(weights_dir / "refiner_det"), **kwargs)
    return {"stage1": stage1, "stage2": stage2}


def predict_cascade(image: np.ndarray, models):
    """1단계가 날짜를 냈으면 그대로, NONE일 때만 YOLO 단계로 다시 찾는다 (식품 500장 420 → 427, 추가 시간은 NONE인 사진뿐)."""
    ymd = rt.predict_image(image, models["stage1"])
    if tuple(ymd) != NONE_DATE:
        return ymd
    stage2 = models["stage2"]
    fields = stage2["yolo"](image)
    if not fields:
        return NONE_DATE
    quads = [refine_quad(image, field["xyxy"], stage2["refiner"])[0] for field in fields]
    records = rt.recognize_boxes(image, quads, stage2, "yolo")
    expiry = [record for record, field in zip(records, fields) if verify_role(field["cls"], record["text"]) == "expiry"]
    winner = rt.choose(expiry) if expiry else None
    return winner["date"] if winner else NONE_DATE
