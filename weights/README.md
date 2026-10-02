# Offline model weights

로컬에는 다음 아홉 모델 디렉터리를 둡니다.

**1단계 (모든 사진)**
- `det_single/`: 4차 파인튜닝 단일 DET (소비기한 줄 검출)
- `rec_v5/`: 4차 파인튜닝 `korean_PP-OCRv5_mobile_rec`
- `rec_v6/`: 4차 파인튜닝 `PP-OCRv6_small_rec`
- `textline_ori/`: `PP-LCNet_x0_25_textline_ori` 방향 분류기
- `uvdoc/`: `UVDoc` 원근 보정 모델 (재시도 때만)

**2단계 (1단계가 NONE일 때만)**
- `yolo_field/best.onnx`: YOLO11n 필드 검출기 (`expiry_field`, `lot_field`), 입력 1280, ONNX
- `refiner_det/`: 범용 `PP-OCRv6_small_det` (YOLO 박스 안 글자 줄 위치 보정)
- `rec_v5_med/`: 5차 파인튜닝 `korean_PP-OCRv5_mobile_rec` (식품 + 의약품)
- `rec_v6_med/`: 5차 파인튜닝 `PP-OCRv6_small_rec` (식품 + 의약품)

Paddle 모델 디렉터리는 `inference.json`, `inference.pdiparams`, `inference.yml`을, `uvdoc/`는
`model.safetensors`, `config.json`, `preprocessor_config.json`, `inference.yml`을 포함해야 합니다.
가중치는 Git에 커밋하지 않고 Release Asset `itda-ocr-weights-v5.tar.gz`로 배포합니다. Asset 내부 경로는
위 디렉터리 이름 그대로입니다. `SHA256SUMS`에는 압축본과 동일한 필수 파일만 기록합니다.
