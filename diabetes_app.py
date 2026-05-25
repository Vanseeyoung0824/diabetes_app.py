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
    st.title("당뇨병 예측 앱 (Streamlit)")
    st.write("입력값으로 당뇨 여부를 예측합니다. 모델이 기대하는 입력 차원에 유의하세요.")

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

    st.sidebar.header("입력 (원본 8개 특성)")
    pregnancies = st.sidebar.number_input("임신횟수", min_value=0, max_value=20, value=0, step=1)
    glucose = st.sidebar.number_input("혈당", min_value=0.0, max_value=300.0, value=120.0, step=0.1)
    blood_pressure = st.sidebar.number_input("혈압", min_value=0.0, max_value=200.0, value=70.0, step=0.1)
    skin_thickness = st.sidebar.number_input("피부두께", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
    insulin = st.sidebar.number_input("인슐린", min_value=0.0, max_value=900.0, value=79.0, step=0.1)
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.sidebar.number_input("가족력", min_value=0.0, max_value=5.0, value=0.5, step=0.01)
    age = st.sidebar.number_input("나이", min_value=1, max_value=120, value=30, step=1)

    # 입력값 원본 8개 특성만 받지만, 모델에는 내부적으로 파생 특성 4개를 추가하여 전달합니다.
    if st.sidebar.button("예측 실행"):
        raw_input = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
        raw_columns = ["임신횟수", "혈당", "혈압", "피부두께", "인슐린", "BMI", "가족력", "나이"]
        raw_df = pd.DataFrame(raw_input, columns=raw_columns)

        # 사용자에게는 입력값 원본 8개만 표시
        display_df = raw_df.copy()

        # 모델 예측을 위해 학습 시 사용된 12개 특성을 생성하여 전달
        blood_health = glucose + blood_pressure
        pancreas_health = glucose + insulin
        cardiovascular_risk = glucose + bmi
        elderly = 1 if age >= 50 else 0

        input_df = pd.DataFrame(
            [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age,
              blood_health, pancreas_health, cardiovascular_risk, elderly]],
            columns=['임신횟수', '혈당', '혈압', '피부두께', '인슐린', 'BMI', '가족력', '나이',
                     '혈액상태', '췌장상태', '심혈관위험지수', '고령']
        )

        try:
            input_scaled = scaler.transform(input_df)
            predicted = log_model_eng.predict(input_scaled)
            prob = None
            if hasattr(log_model_eng, "predict_proba"):
                prob = log_model_eng.predict_proba(input_scaled)
        except Exception as exc:
            st.error(f"예측 중 오류가 발생했습니다: {exc}")
            st.write("참고: 모델이 기대하는 특성 순서와 수가 로드된 스케일러/모델과 일치하는지 확인하세요.")
            return

        result_label = "당뇨" if int(predicted[0]) == 1 else "정상"
        probability = (prob[0][1] * 100) if prob is not None else None

        st.subheader("예측 결과")
        st.metric(label="진단", value=result_label)
        if probability is not None:
            st.write(f"당뇨 확률: {probability:.1f}%")
        else:
            st.write("모델이 확률을 제공하지 않습니다.")

        st.write("### 입력값 (표시용)")
        st.dataframe(display_df)


if __name__ == "__main__":
    main()
