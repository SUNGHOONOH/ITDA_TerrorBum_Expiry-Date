# REC weights

로컬에는 다음 두 export 디렉터리를 둡니다.

- `det_single/`: 현재 임시 기본값인 단일 DET
- `rec_v5/`: `korean_PP-OCRv5_mobile_rec`
- `rec_v6/`: `PP-OCRv6_small_rec`

2단계 DET를 선택하면 `det_full/`과 `det_date/`를 추가하고 `ITDA_DET_MODE=cascade`로 실행합니다.

각 디렉터리는 `inference.json`, `inference.pdiparams`, `inference.yml`을 포함해야 합니다.
`*.pdiparams`는 GitHub 단일 파일 제한과 재현성을 고려해 Git에 커밋하지 않고 Release Asset
`itda-ocr-weights.tar.gz`로 배포합니다. Asset 내부 경로는 `det_single/...`, `rec_v5/...`,
`rec_v6/...`여야 합니다.
