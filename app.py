import streamlit as st
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 페이지 설정
st.set_page_config(
    page_title="당뇨병 위험도 예축 시스템",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 커스텀 CSS
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #333;
    }
    .subtitle {
        text-align: center;
        font-size: 14px;
        color: #666;
        margin-bottom: 30px;
    }
    .prediction-button {
        display: flex;
        justify-content: center;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .result-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 15px;
        color: #333;
    }
    .normal-result {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        color: #155724;
        margin-top: 10px;
    }
    .abnormal-result {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 15px;
        color: #721c24;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 제목
st.markdown('<div class="main-title">🏥 당뇨병 위험도 예축 시스템</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">아래 정보를 입력하시면 AI 모델이 당뇨병 가능성을 예측합니다.</div>', unsafe_allow_html=True)

st.markdown("---")

# 모델 및 스케일러 로드/생성
def load_or_create_model():
    model_path = "diabetes_model.pkl"
    scaler_path = "scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    else:
        # 샘플 데이터로 모델 학습
        # 8개 feature를 가진 더미 데이터 생성
        np.random.seed(42)
        n_samples = 200
        X = np.random.randn(n_samples, 8) * [5, 30, 20, 15, 100, 10, 0.5, 15] + [2, 120, 70, 20, 100, 25, 0.5, 40]
        y = np.random.randint(0, 2, n_samples)
        
        # 스케일러 생성 및 데이터 스케일
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 모델 학습
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # 모델과 스케일러 저장
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
    
    return model, scaler

# 입력 필드 구성 - 2열 레이아웃
col1, col2 = st.columns(2)

with col1:
    st.subheader("건강 정보")
    pregnancies = st.number_input("입실 횟수", min_value=0, max_value=20, value=2, step=1)
    glucose = st.number_input("혈당 (Glucose)", min_value=0, max_value=200, value=112, step=1)
    blood_pressure = st.number_input("혈압 (Blood Pressure)", min_value=0, max_value=200, value=79, step=1)
    skin_thickness = st.number_input("피부 두께 (Skin Thickness)", min_value=0, max_value=100, value=34, step=1)

with col2:
    st.subheader("인슐린 & 기타")
    insulin = st.number_input("인슐린 (Insulin)", min_value=0, max_value=900, value=75, step=1)
    bmi = st.number_input("BMI (체질량지수)", min_value=0.0, max_value=60.0, value=25.70, step=0.01)
    dpf = st.number_input("당뇨 내력 기존치 (DPF)", min_value=0.0, max_value=2.5, value=0.470, step=0.001)
    age = st.number_input("나이", min_value=0, max_value=120, value=36, step=1)

st.markdown("---")

# 예측 버튼
col_button = st.columns([1, 1, 1])
with col_button[1]:
    if st.button("🏥 당뇨병 여부 예측하기", use_container_width=True, key="predict_btn"):
        # 모델 로드
        model, scaler = load_or_create_model()
        
        # 입력 데이터 준비
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                insulin, bmi, dpf, age]])
        
        # 데이터 스케일
        input_scaled = scaler.transform(input_data)
        
        # 예측
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # 결과 표시
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">👤 예축 결과</div>', unsafe_allow_html=True)
        
        if prediction == 0:
            result_text = f"✅ 정상 범주로 예측됩니다. (당뇨 가능성: {probability[1]*100:.1f}%)"
            st.markdown(f'<div class="normal-result">{result_text}</div>', unsafe_allow_html=True)
        else:
            result_text = f"⚠️ 당뇨병 위험군으로 예측됩니다. (당뇨 가능성: {probability[1]*100:.1f}%)"
            st.markdown(f'<div class="abnormal-result">{result_text}</div>', unsafe_allow_html=True)
        
        # 진행도 바
        st.progress(probability[1])
        
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
    <div style="text-align: center; font-size: 12px; color: #999; margin-top: 20px;">
    ⚠️ 본 예측 시스템은 의료 진단을 대체할 수 없습니다. 정확한 진단을 위해 의료 전문가와 상담하세요.
    </div>
""", unsafe_allow_html=True)
