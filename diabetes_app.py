import joblib
from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


@st.cache_resource
def load_scaler(path: Path):
    return joblib.load(path)


def main() -> None:
    st.title("당뇨병 예측 앱")
    st.write("훈련된 분류 모델과 동일한 전처리를 사용하여 입력값 기반 당뇨병 여부를 예측합니다.")

    base_path = Path(__file__).resolve().parent
    model_path = base_path / "diabetes_model2.pkl"
    scaler_path = base_path / "diabetes_scaler.pkl"

    try:
        log_model_eng = load_model(model_path)
        scaler = load_scaler(scaler_path)
    except FileNotFoundError as exc:
        st.error(f"모델 또는 스케일러 파일을 찾을 수 없습니다: {exc}")
        return
    except Exception as exc:
        st.error(f"파일 로드 중 오류가 발생했습니다: {exc}")
        return

    with st.form(key="diabetes_form"):
        pregnancies = st.number_input("임신횟수", min_value=0, max_value=20, value=0, step=1)
        glucose = st.number_input("혈당", min_value=0.0, max_value=300.0, value=120.0, step=0.1)
        blood_pressure = st.number_input("혈압", min_value=0.0, max_value=200.0, value=70.0, step=0.1)
        skin_thickness = st.number_input("피부두께", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
        insulin = st.number_input("인슐린", min_value=0.0, max_value=900.0, value=79.0, step=0.1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        dpf = st.number_input("가족력", min_value=0.0, max_value=5.0, value=0.5, step=0.01)
        age = st.number_input("나이", min_value=1, max_value=120, value=30, step=1)

        submit_button = st.form_submit_button("예측 실행")

    if submit_button:
        blood_health = glucose + blood_pressure
        pancreas_health = glucose + insulin
        cardiovascular_risk = glucose + bmi
        elderly = 1 if age >= 50 else 0

        input_data = pd.DataFrame(
            [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age,
              blood_health, pancreas_health, cardiovascular_risk, elderly]],
            columns=[
                "임신횟수", "혈당", "혈압", "피부두께", "인슐린", "BMI", "가족력", "나이",
                "혈액상태", "췌장상태", "심혈관위험지수", "고령"
            ]
        )

        try:
            input_data_scaled = scaler.transform(input_data)
            predicted = log_model_eng.predict(input_data_scaled)
            prob = log_model_eng.predict_proba(input_data_scaled)
        except Exception as exc:
            st.error(f"예측 중 오류가 발생했습니다: {exc}")
            return

        result_label = "당뇨" if predicted[0] == 1 else "정상"
        probability = prob[0][1] * 100

        st.subheader("예측 결과")
        st.metric(label="진단", value=result_label)
        st.write(f"당뇨 확률: {probability:.1f}%")
        st.write("### 입력값 요약")
        st.dataframe(input_data)


if __name__ == "__main__":
    main()
