# Custom data

`custom_data_v2_original_validation.zip` Release Asset (또는 외부 다운로드 링크)에 500장의 검수 완료 자체 수집 학습 이미지, CVAT box/transcription XML, DET 라벨, REC 라벨과 별도 validation 214장 원본 JPEG를 포함한다. 직접 촬영 학습 이미지 500장은 기존 데이터와 함께 `source sampling`(데이터 출처별 샘플링)으로 파인튜닝에 포함했다. 학습용 500장 원본은 JPEG 최대 긴 변 1600px로 압축했고 XML 좌표도 같은 비율로 변환했다.

`validation_214/`은 학습에 사용하지 않은 별도 실사 검증 214장과 `labels.csv`를 포함한다. 원본 214장 중 exact `NONE` 14장을 제외한 200장을 CPU 테스트용 직접촬영 subset으로 사용하며, 부분 라벨은 제외하지 않는다. 파일별 출처·제품군·source sampling 정보는 이 저장소의 `metadata/` CSV에 남긴다.
