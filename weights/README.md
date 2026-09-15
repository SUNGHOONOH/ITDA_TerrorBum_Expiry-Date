# Offline model weights

로컬에는 다음 다섯 모델 디렉터리를 둡니다.

- `det_single/`: 4차 파인튜닝 단일 DET 후보
- `rec_v5/`: 4차 파인튜닝 `korean_PP-OCRv5_mobile_rec`
- `rec_v6/`: 4차 파인튜닝 `PP-OCRv6_small_rec`
- `textline_ori/`: `PP-LCNet_x0_25_textline_ori` 방향 분류기
- `uvdoc/`: `UVDoc` 원근 보정 모델

앞의 네 디렉터리는 `inference.json`, `inference.pdiparams`, `inference.yml`을 포함해야
하며, `uvdoc/`는 `model.safetensors`, `config.json`, `preprocessor_config.json`,
`inference.yml`을 포함해야 합니다.
`*.pdiparams`는 GitHub 단일 파일 제한과 재현성을 고려해 Git에 커밋하지 않고 Release Asset
`itda-ocr-weights-v4.tar.gz`로 배포합니다. Asset 내부 경로는 `det_single/...`, `rec_v5/...`,
`rec_v6/...`, `textline_ori/...`, `uvdoc/...`여야 합니다. `SHA256SUMS`에는 압축본과 동일한
필수 파일만 기록하며, Hugging Face 캐시 부산물은 포함하지 않습니다.
