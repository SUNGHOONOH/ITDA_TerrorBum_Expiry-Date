# ITDA TerrorBum Expiry-Date

제3회 ITDA 연합학술제 — 상품 소비기한 OCR 추출

## 환경
- Python 3.10
- CPU 전용 (GPU 불필요)

## 설치
```bash
pip install -r requirements.txt
```

## 가중치
`weights/` 안에 있으면 우선 사용합니다. 현재 확정된 인식기는 항상 두 개를 함께 사용합니다.

- `weights/rec_v5/`: 파인튜닝한 `korean_PP-OCRv5_mobile_rec`
- `weights/rec_v6/`: 파인튜닝한 `PP-OCRv6_small_rec`
- `weights/det_single/`: 현재 임시 기본값인 단일 PP-OCRv6 DET
- `weights/det_full/`, `weights/det_date/`: 2단계 DET를 선택할 때 교체·추가
- `weights/`의 `*.pdiparams`는 Git에 올리지 않고 GitHub Release Asset으로 배포합니다.

채점 서버는 인터넷이 차단되어 있습니다. 운영진이 온라인 상태에서 **채점 전에 한 번** 아래처럼 실행해 가중치를 내려받습니다.
```bash
export ITDA_PP_OCR_WEIGHTS_URL='https://github.com/<org>/<repo>/releases/download/<tag>/itda-ocr-weights.tar.gz'
bash download_weights.sh
```
`predict.ipynb`는 인터넷을 사용하지 않으며, 이미 `weights/`에 내려받은 모델만 읽습니다. 가중치 다운로드 코드는 `download_weights.sh`에만 있습니다.

## 실행
```bash
export ITDA_INPUT_DIR=./val_images
export ITDA_OUTPUT_PATH=./submission.csv
jupyter nbconvert --to notebook --execute predict.ipynb --output /tmp/out.ipynb
```

## 출력
`submission.csv` — `image_id, year, month, day, final_date`
미인식 필드는 `NONE`입니다. 전 필드 미인식 시 `final_date`도 `NONE`이며, 일부만 인식된 경우에는 없는 필드를 포함해 하이픈으로 연결합니다(예: `NONE-02-18`).

## 추론 구조
1. DET 후보 영역 생성 (full/date-only 선택 전까지 임시 어댑터)
2. 각 후보를 파인튜닝한 한국어 PP-OCRv5와 PP-OCRv6로 항상 함께 인식
3. 두 인식 결과를 날짜 후보·신뢰도·소비/유통기한 문맥으로 결합
4. 날짜 형식·달력 유효성·소비/유통기한 문맥으로 최종 선택
5. 불확실한 결과만 원본 ROI를 확대하고 CLAHE/도트 획 보정 후 재인식

현재 기본 `ITDA_DET_MODE=single`은 기존 단일 PP-OCRv6 DET입니다. `full`, `date`, `cascade`를 선택하면 해당 export 디렉터리를 사용하므로 DET 결정을 바꿔도 REC·parser 코드는 바뀌지 않습니다. REC 두 개를 고르는 구조가 아니라, 두 REC를 `main` 경로에서 매번 함께 사용하도록 되어 있습니다.

`000001~000200`은 로컬 검증 전용 분리이며, 제출 노트북은 운영진이 `INPUT_DIR`에 넣은 모든 이미지를 처리합니다.

## 자체 수집 데이터
가산점 심사용 자체 수집 데이터는 `custom_data_v1.zip` Release Asset으로 제공한다. `custom_data/`에는 데이터 위치와 구조 설명, 이미지 출처·제품군·source sampling 메타데이터 CSV를 둔다.

## 구조
```
predict.ipynb     메인 추론 노트북 (채점 대상)
predict_runtime.py 공통 CPU 추론 코드
requirements.txt  의존성
weights/          모델 가중치
  det_single/      현재 임시 단일 DET
  det_full/        full DET (선택)
  det_date/        date-only DET (선택)
  rec_v5/         파인튜닝 korean PP-OCRv5 (Release Asset)
  rec_v6/         파인튜닝 PP-OCRv6 (Release Asset)
notebooks/        실험·분석 (채점 대상 아님)
custom_data/      가산점 심사용 자체 수집 이미지·라벨
```
