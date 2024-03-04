# Deepfake-detection
한국인 특화 딥페이크 영상 탐지
## 프로젝트 목적
한국인 딥페이크 영상을 높은 정확도로 판별하는 서비스 개발 
## 프로젝트 내용
- 논문의 XceptionNet 모델을 활용한 딥페이크 연상 판별
- 기학습된 모델에 한국인 딥페이크 데이터셋 추가 학습
- Yolov5를 활용한 얼굴 검출 방식 속도 개선
- 전이학습을 통해 한국인 데이터에 특화되도록 모델 튜닝
- Fast API를 이용한 API 개발
- 딥페이크 탐지 웹서비스 개발
## 작업 기간
2023. 11. 25. ~ 2023. 12. 03
## 사용 모델
Xception(화소 기반), Yolov5(얼굴 검출 방식)
## 사용 언어 및 개발 환경
Python/ PyTorch, NumPy, matplotlib, Streamlit, FastAPI, GCS, Firebase
## PoC
Web
