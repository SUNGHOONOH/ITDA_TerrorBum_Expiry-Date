# ITDA TerrorBum Expiry-Date

오프라인 OCR 제출 저장소입니다. `predict.ipynb`를 처음부터 끝까지 실행하면 `ITDA_INPUT_DIR`의 이미지를 읽어 `ITDA_OUTPUT_PATH`에 `submission.csv`를 저장합니다.

## 제출 구조

```text
predict.ipynb        채점 대상 노트북
requirements.txt      Python 3.10 의존성 고정 목록
README.md             설치·가중치·오프라인 실행 안내
download_weights.sh   가중치 다운로드 전용 스크립트
weights/              로컬 모델 가중치
notebooks/itda_ocr/   노트북이 import하는 추론 모듈
test_imgs/             로컬 10장 smoke test용 이미지 (선택)
```

## 1. 온라인 상태에서 사전 준비

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



위 단계가 끝난 뒤 Wi-Fi 또는 네트워크를 끄십시오. `download_weights.sh`는 온라인 상태에서만 실행합니다.

## 2. 10장 오프라인 smoke test

`test_imgs/`에 테스트 이미지 10장을 넣어 두면 아래 명령 하나로 실행할 수 있습니다. 이 검사는 정확도가 아니라 패키지·가중치·노트북 실행과 CSV 출력 계약을 확인합니다.

```bash
ITDA_INPUT_DIR=./test_imgs ITDA_OUTPUT_PATH=/tmp/submission.csv \
jupyter nbconvert --to notebook --execute predict.ipynb \
  --ExecutePreprocessor.timeout=2400 \
  --output /tmp/executed.ipynb
```

`jupyter`를 찾지 못하면 `jupyter nbconvert` 대신 `python -m jupyter nbconvert`를 사용합니다. `/tmp/submission.csv`가 생성되고 10행이며 열이 `image_id,year,month,day,final_date`이면 smoke test를 통과한 것입니다.

## 3. 전체 검증 또는 제출 전 실행

검증 이미지가 들어 있는 폴더를 지정해 동일한 방식으로 실행합니다. 실제 채점 입력에서는 운영진이 `ITDA_INPUT_DIR`와 `ITDA_OUTPUT_PATH`만 주입합니다.

```bash
ITDA_INPUT_DIR=/path/to/500_images \
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
