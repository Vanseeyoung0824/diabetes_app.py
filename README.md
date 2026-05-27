# 당뇨병 위험도 예측 시스템

사진과 동일한 당뇨병 위험도 예측 Streamlit 앱입니다.

## 설치 및 실행 방법

### 1. 필수 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 2. 앱 실행
```bash
streamlit run app.py
```

또는 Windows에서:
```bash
python -m streamlit run app.py
```

### 3. 웹 브라우저에서 접속
- 자동으로 `http://localhost:8501`에서 앱이 열립니다.

## 기능

- **건강 정보 입력**: 8가지 건강 지표 입력
  - 입실 횟수 (Pregnancies)
  - 혈당 (Glucose)
  - 혈압 (Blood Pressure)
  - 피부 두께 (Skin Thickness)
  - 인슐린 (Insulin)
  - BMI (체질량지수)
  - 당뇨 내력 기존치 (DPF - Diabetes Pedigree Function)
  - 나이 (Age)

- **예측 기능**: 입력된 정보를 바탕으로 AI 모델이 당뇨병 위험도를 예측
- **결과 표시**: 예측 결과와 확률을 시각적으로 표시

## 배포

### Streamlit Community Cloud에 배포
1. GitHub 저장소에 코드 푸시
2. [Streamlit Community Cloud](https://streamlit.io/cloud) 방문
3. 새 앱 생성 및 배포

### 기타 배포 방법
- Heroku, AWS, Azure 등 다양한 클라우드 플랫폼에 배포 가능

## 주의사항

⚠️ 본 예측 시스템은 의료 진단을 대체할 수 없습니다. 정확한 진단을 위해 의료 전문가와 상담하세요.
