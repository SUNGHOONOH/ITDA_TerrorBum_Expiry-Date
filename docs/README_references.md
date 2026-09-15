## 참고문헌

요약서 PDF 본문의 [1]~[10] 번호와 대응한다.

[1] V. Florea, T. Rebedea, "Expiry date recognition using deep neural networks," *International Journal of User-System Interaction*, 13(1), pp. 1–17, 2020. doi:10.37789/ijusi.2020.13.1.1 · https://rochi.utcluj.ro/ijusi/articles/IJUSI-13-1-Florea.pdf
— 검출기·인식기 재학습 비교(Table 2: 인식기만 실사 재학습 34.8% > 검출기만 24.2%), 날짜 후보 중 가장 미래 날짜 선택.

[2] A. C. Seker, S. C. Ahn, "A generalized framework for recognition of expiration dates on product packages using fully convolutional networks," *Expert Systems with Applications*, vol. 203, 117310, 2022. doi:10.1016/j.eswa.2022.117310 · ExpDate 데이터셋: https://felizang.github.io/expdate/
— 13가지 날짜 형식, 소비기한·제조일·코드 표시 구분, 날짜 검출 후 일·월·연 검출의 2단계 구조.

[3] C. Cui et al., "PaddleOCR 3.0 Technical Report," arXiv:2507.05595, 2025. https://arxiv.org/abs/2507.05595 · PaddleOCR 공식 문서: https://www.paddleocr.ai
— 글줄 원근 보정·방향 분류 단계, 파인튜닝 데이터 비율(기존:추가 10:1~5:1, 공식 문서), 경량·대형 모델 CPU 속도.

[4] Y. Zhang et al., "PP-OCRv6: From 1.5M to 34.5M Parameters, Surpassing Billion-Scale VLMs on OCR Tasks," arXiv:2606.13108, 2026. https://arxiv.org/abs/2606.13108
— PP-OCRv6 small 검출기·인식기.

[5] F. Verhoeven, T. Magne, O. Sorkine-Hornung, "UVDoc: Neural Grid-based Document Unwarping," *SIGGRAPH Asia 2023 Conference Papers*, 2023. doi:10.1145/3610548.3618174
— 곡면 펴기.

[6] K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," in P. S. Heckbert (ed.), *Graphics Gems IV*, Academic Press, pp. 474–485, 1994. doi:10.1016/B978-0-12-336156-1.50061-6
— 대비 향상(CLAHE).

[7] 식품의약품안전처, 『식품등의 표시기준』 고시 제2025-60호(2025.8.29.). https://www.mfds.go.kr/brd/m_211/view.do?seq=14917
— 소비기한·제조일 등 표시 기준, 한글 표시사항.

[8] 식품의약품안전처 수입식품정보마루, 「수입신고서 작성 요령 Q&A」, https://impfood.mfds.go.kr/CFBCC05F01 · 「수입식품등의 수입신고서」 서식(유통기한란), https://law.go.kr/flDownload.do?flSeq=122255183 · 공공데이터포털 「식품의약품안전처_수입식품 수입신고별 한글표시사항」, https://www.data.go.kr/data/15110213/openapi.do
— 수입신고서의 기한 기재 항목, 수입신고별 한글표시사항 공개 데이터.

[9] Codex Alimentarius, *General Standard for the Labelling of Prepackaged Foods* (CXS 1-1985) · GS1 Application Identifiers 11(제조일)·13(포장일)·15(품질유지기한)·16(판매기한)·17(소비기한), https://ref.gs1.org/ai/
— 국제 날짜 표시 정의, 바코드 속 날짜 정보.

[10] J. Baek, Y. Matsui, K. Aizawa, "What If We Only Use Real Datasets for Scene Text Recognition? Toward Scene Text Recognition With Fewer Labels," *CVPR*, pp. 3113–3122, 2021.
— 합성 데이터 없이 실제 데이터만으로 학습하는 효과.
