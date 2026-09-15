# ITDA TerrorBum Expiry-Date

제3회 ITDA 연합학술제 상품 소비기한 OCR 제출본이다. 운영진이 `predict.ipynb`을 Run All 하면, 주입된 이미지 폴더를 처리해 `submission.csv`를 생성한다.

## 모델 구성

- 단일 파인튜닝 `PP-OCRv6_small_det`
- 파인튜닝 `korean_PP-OCRv5_mobile_rec`와 `PP-OCRv6_small_rec`의 동시 인식
- 텍스트 방향 분류, CLAHE/패딩 rescue, 로컬 UVDoc 원근 보정
- 소비/유통기한 문맥, 날짜 유효성, 좌표를 결합하는 날짜 parser

GPU와 외부 API를 사용하지 않는다. CPU 스레드는 4개로 고정한다.

## 가중치 준비

채점 서버는 인터넷이 차단되어 있다. 운영진은 온라인 상태에서 **한 번만** 다음을 실행해 Release Asset의 가중치를 `weights/`에 내려받는다.

```bash
bash download_weights.sh
```

`predict.ipynb`에는 가중치 다운로드 코드가 없으며, 이미 존재하는
`weights/det_single`, `weights/rec_v5`, `weights/rec_v6`, `weights/textline_ori`,
`weights/uvdoc`만 읽는다.

## 채점 재현성 검증

공지의 자가 점검 절차와 같다. 가중치 설치가 끝난 뒤 인터넷을 끊고 아래 명령을 실행한다.

```bash
git clone https://github.com/SUNGHOONOH/ITDA_TerrorBum_Expiry-Date.git
cd ITDA_TerrorBum_Expiry-Date
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

# 온라인 상태에서 1회 실행
bash download_weights.sh

# 인터넷을 차단한 뒤 실행
ITDA_INPUT_DIR=/tmp/test_imgs ITDA_OUTPUT_PATH=/tmp/submission.csv \
  jupyter nbconvert --to notebook --execute predict.ipynb \
  --ExecutePreprocessor.timeout=2400 --output /tmp/executed.ipynb
```

`/tmp/submission.csv`가 생성되면 제출 환경 검증을 통과한다.

## 4차 가중치 교체

현재 `ft2-v0`는 제출 구조를 검증하기 위한 fallback이다. 4차 학습이 끝나면 동일한 export 형식으로 다음 세 학습 모델 디렉터리의 내용만 교체한다.

```text
weights/det_single/   4차 단일 DET export
weights/rec_v5/       4차 PP-OCRv5 export
weights/rec_v6/       4차 PP-OCRv6 export
```

`weights/textline_ori/`와 `weights/uvdoc/`는 4차 전처리용 고정 모델이라 그대로 유지한다.

그 뒤 다섯 디렉터리를 포함한 `itda-ocr-weights.tar.gz`로 새 Release Asset을 만들고,
`download_weights.sh`의 `WEIGHTS_URL`을 새 Release URL로 바꿔 커밋한다.
`predict.ipynb`, `itda_ocr/`, `requirements.txt`는 export 형식과 입력·출력 계약이 같다면 수정하지 않는다.

## 출력

`submission.csv` 컬럼은 `image_id,year,month,day,final_date`다. 날짜를 찾지 못하면 모든 날짜 필드와 `final_date`를 `NONE`으로 기록한다.

## 자체 수집 데이터

가산점 심사용 `custom_data_v1.zip` Release Asset에는 검수 완료 자체 수집 이미지 500장, CVAT box/transcription XML, DET·REC 라벨을 담는다. 별도 실사 검증 214장은 학습에 사용하지 않았으며, 파일 출처·제품군·source sampling 정보는 `custom_data/metadata/`에 기록되어 있다.

## 문서

요약서 PDF의 논문 인용 번호 `[1]`~`[10]` 대응표는 [docs/README_references.md](docs/README_references.md)에 있다. 요약서 PDF 자체는 안내된 이메일 제출물이다.

## 구조

```text
predict.ipynb          채점 대상 메인 노트북
itda_ocr/              오프라인 추론·parser 모듈
download_weights.sh    온라인 상태의 사전 가중치 다운로드 전용
requirements.txt       Python 3.10 고정 의존성
weights/               Release Asset이 풀리는 위치
custom_data/           자체 수집 데이터 설명·메타데이터
docs/                  요약서 인용 참고문헌
```
