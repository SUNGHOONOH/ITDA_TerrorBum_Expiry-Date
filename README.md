# ITDA TerrorBum Expiry-Date

식품 포장 사진에서 소비기한 날짜를 읽어 제출 형식으로 변환하는 오프라인 OCR 프로젝트입니다. `predict.ipynb`를 실행하면 입력 이미지에서 날짜를 추출해 `submission.csv`로 저장합니다.

| 구분 | 내용 |
|---|---|
| 내 작업 | OCR 오류 유형 분석, 이미지 전처리·재시도 경로, 날짜 후보 후처리와 검증 |
| 사용 기술 | Python · PaddleOCR · 이미지 전처리 · 날짜 파싱 · 오류 분석 |
| 실행 조건 | 인터넷이 차단된 채점 환경 · CPU 추론 · 가중치 사전 준비 |

## 문제와 내 역할

포장 사진에는 소비기한 외에도 제조일자·로트번호 등 비슷하게 생긴 숫자가 있고, 인쇄가 흐리거나 기울어지면 OCR이 날짜를 분리하거나 잘못 읽습니다. 글자를 인식하는 것만으로 끝내지 않고 **어느 날짜를 최종 답으로 택할지**와 **읽지 못할 때 어떻게 처리할지**를 함께 설계했습니다. 제가 집중한 부분은 오류 사례를 분류하고 입력 보정·후처리·검증 경로를 다듬는 일이었습니다.

## 선택한 방법과 이유

| 고민 | 선택한 방법 | 확인할 점 |
|---|---|---|
| 모든 이미지에 무거운 보정을 적용해야 하나? | 기본 OCR 경로로 먼저 읽고, 결과가 없거나 신뢰도가 낮을 때 대비·여백·문서 보정을 재시도 | 오프라인 CPU에서 모든 입력에 동일한 비용을 들이지 않음 |
| 숫자가 날짜처럼 보이면 바로 채택해도 되나? | 날짜 형식과 달력 유효성, 주변의 날짜 역할 단서를 같이 판단 | 제조일자·로트번호와 혼동 가능성은 남음 |
| 식별할 수 없는 이미지는? | 추측한 날짜 대신 `NONE`을 기록 | 잘못된 확정 값을 줄이는 대신 미인식이 발생할 수 있음 |

## 확인한 결과와 한계

개인적으로 구성한 고정 검증 세트 200장 중 187장에서 날짜를 맞혀 **93.5%**를 확인했습니다. 이 수치는 공모전 비공개 평가 점수나 실제 서비스 성능이 아닙니다. 촬영 조건과 상품 종류가 바뀐 데이터로 추가 검증해야 하며, 실행 시간도 공식 성능 지표로 제시하지 않습니다.

## 주요 파일 안내

| 경로 | 확인할 내용 |
|---|---|
| [`predict.ipynb`](predict.ipynb) | 입력 이미지에서 제출 CSV까지 이어지는 전체 실행 |
| [`notebooks/itda_ocr/runtime.py`](notebooks/itda_ocr/runtime.py) | OCR 경로와 저신뢰 결과의 재시도 |
| [`notebooks/itda_ocr/parser.py`](notebooks/itda_ocr/parser.py) | 날짜 후보 정규화·선택·거절 |
| [`download_weights.sh`](download_weights.sh) | 오프라인 실행 전 가중치 준비 |

## 실행 방법

다음은 기존 제출 환경에서 사용한 설치·실행 절차입니다. `ITDA_INPUT_DIR`의 이미지를 읽고 `ITDA_OUTPUT_PATH`에 `submission.csv`를 저장합니다. 학습·검증 원본 이미지는 이 저장소에 포함하지 않습니다.

### 제출 구조

```text
predict.ipynb        채점 대상 노트북
requirements.txt      Python 3.10 의존성 고정 목록
README.md             설치·가중치·오프라인 실행 안내
download_weights.sh   가중치 다운로드 전용 스크립트
weights/              로컬 모델 가중치
notebooks/itda_ocr/   노트북이 import하는 추론 모듈
test_imgs/             로컬 10장 smoke test용 이미지 (선택)
```

### 1. 온라인 상태에서 사전 준비

채점 서버는 인터넷이 차단되어 있으므로, 아래 설치와 가중치 다운로드를 인터넷 연결 상태에서 한 번만 실행합니다. 이미 `.venv310`을 사용 중이면 새 가상환경을 만들지 말고 해당 환경을 활성화하면 됩니다.

```bash
# 새로 clone할 때
git clone https://github.com/SUNGHOONOH/ITDA_TerrorBum_Expiry-Date.git
cd ITDA_TerrorBum_Expiry-Date

# 이미 clone한 저장소라면 위 두 줄 대신 아래처럼 루트로 이동
# cd /path/to/ITDA_TerrorBum_Expiry-Date

# 기존 환경을 사용할 때 (.venv310이 이미 있으면 이 줄)
. .venv310/bin/activate
# .venv310이 저장소 상위 폴더에 있으면 위 줄 대신
# . ../.venv310/bin/activate

# 새 환경을 만들 때는 위 줄 대신 다음 두 줄
# python3.10 -m venv .venv
# . .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
bash download_weights.sh

# 가중치 파일과 해시 확인 (Linux)
(cd weights && sha256sum -c SHA256SUMS)
# macOS에서는 위 명령 대신 다음을 사용
# (cd weights && shasum -a 256 -c SHA256SUMS)
```

`download_weights.sh`가 준비하는 모델은 `det_single`, `rec_v5`, `rec_v6`, `textline_ori`, `uvdoc`입니다. `predict.ipynb`와 `notebooks/itda_ocr/`에는 실행 중 다운로드하는 코드가 없습니다. 가중치가 없거나 해시가 다르면 즉시 오류가 나도록 되어 있습니다.

### 2. 전체 검증 또는 제출 전 실행

검증 이미지가 들어 있는 폴더를 지정해 동일한 방식으로 실행합니다. 실제 채점 입력에서는 운영진이 `ITDA_INPUT_DIR`와 `ITDA_OUTPUT_PATH`만 주입합니다.

```bash
ITDA_INPUT_DIR=/tmp/test_images \
ITDA_OUTPUT_PATH=/tmp/submission.csv \
jupyter nbconvert --to notebook --execute predict.ipynb \
  --ExecutePreprocessor.timeout=2400 \
  --output /tmp/executed.ipynb
```


최종 셀은 반드시 다음 형식으로 인덱스 없이 저장합니다.

```python
df.to_csv(OUTPUT_PATH, index=False)
```

출력 열은 `image_id,year,month,day,final_date`입니다. 날짜를 찾지 못한 경우 날짜 필드와 `final_date`를 `NONE`으로 기록합니다.

## 참고문헌

아래 번호는 제출 요약서의 `[1]`~`[10]` 인용 번호와 대응합니다.

[1] V. Florea, T. Rebedea, “Expiry date recognition using deep neural networks,” *International Journal of User-System Interaction*, 13(1), 2020. [논문](https://rochi.utcluj.ro/ijusi/articles/IJUSI-13-1-Florea.pdf)

[2] A. C. Seker, S. C. Ahn, “A generalized framework for recognition of expiration dates on product packages using fully convolutional networks,” *Expert Systems with Applications*, 203, 117310, 2022. [논문·ExpDate 데이터셋](https://felizang.github.io/expdate/)

[3] C. Cui et al., “PaddleOCR 3.0 Technical Report,” arXiv:2507.05595, 2025. [논문](https://arxiv.org/abs/2507.05595) · [공식 문서](https://www.paddleocr.ai)

[4] Y. Zhang et al., “PP-OCRv6: From 1.5M to 34.5M Parameters, Surpassing Billion-Scale VLMs on OCR Tasks,” arXiv:2606.13108, 2026. [논문](https://arxiv.org/abs/2606.13108)

[5] F. Verhoeven, T. Magne, O. Sorkine-Hornung, “UVDoc: Neural Grid-based Document Unwarping,” *SIGGRAPH Asia*, 2023. [DOI](https://doi.org/10.1145/3610548.3618174)

[6] K. Zuiderveld, “Contrast Limited Adaptive Histogram Equalization,” in *Graphics Gems IV*, 1994. [DOI](https://doi.org/10.1016/B978-0-12-336156-1.50061-6)

[7] 식품의약품안전처, 『식품등의 표시기준』 고시 제2025-60호. [고시](https://www.mfds.go.kr/brd/m_211/view.do?seq=14917)

[8] 식품의약품안전처 수입식품정보마루, 「수입신고서 작성 요령」. [수입식품정보마루](https://impfood.mfds.go.kr/CFBCC05F01) · [법령정보 서식](https://law.go.kr/flDownload.do?flSeq=122255183) · [공공데이터포털](https://www.data.go.kr/data/15110213/openapi.do)

[9] Codex Alimentarius, *General Standard for the Labelling of Prepackaged Foods* (CXS 1-1985) · GS1 Application Identifiers. [GS1 날짜 식별자](https://ref.gs1.org/ai/)

[10] J. Baek, Y. Matsui, K. Aizawa, “What If We Only Use Real Datasets for Scene Text Recognition? Toward Scene Text Recognition With Fewer Labels,” *CVPR*, 2021. [논문](https://openaccess.thecvf.com/content/CVPR2021/papers/Baek_What_if_We_Only_Use_Real_Datasets_for_Scene_Text_CVPR_2021_paper.pdf)
