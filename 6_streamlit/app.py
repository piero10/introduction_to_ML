import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import numpy as np

# ============================
# Настройка страницы
# ============================
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

# ============================
# Загрузка данных и модели
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv("data/titanic.csv")
    return df

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

df = load_data()
model = load_model()

#st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")
st.title("🚢 Titanic Survival Predictor")

tabs = st.tabs(["📊 Аналитика", "🔮 Предсказание", "🧠 О модели"])

# ============================
# TAB 1: Аналитика
# ============================
with tabs[0]:
    st.subheader("Обзор данных")

    st.dataframe(df.head())

    c1, c2 = st.columns(2)

    with c1:
        fig_sex = px.histogram(df, x="Sex", color="Survived",
                               title="Выживаемость по полу", barmode="group")
        st.plotly_chart(fig_sex, use_container_width=True)

    with c2:
        fig_class = px.histogram(df, x="Pclass", color="Survived",
                                 title="Выживаемость по классу билета", barmode="group")
        st.plotly_chart(fig_class, use_container_width=True)

    st.markdown("### 📈 Возраст и выживаемость")
    fig_age = px.histogram(df, x="Age", color="Survived", nbins=30, title="Возраст и шанс выживания")
    st.plotly_chart(fig_age, use_container_width=True)

    st.markdown("### 💰 Стоимость билета (Fare) и выживаемость")
    fig_fare = px.box(df, x="Survived", y="Fare", points="all", title="Стоимость билета и выживаемость")
    st.plotly_chart(fig_fare, use_container_width=True)

# ============================
# TAB 2: Предсказание
# ============================
with tabs[1]:
    st.subheader("Введите данные пассажира")

    c1, c2, c3 = st.columns(3)

    with c1:
        pclass = st.selectbox("Класс билета", [1, 2, 3], index=2)
        sex = st.selectbox("Пол", ["male", "female"])
    with c2:
        age = st.slider("Возраст", 1, 80, 30)
        fare = st.slider("Стоимость билета", 0.0, 500.0, 50.0)
    with c3:
        embarked = st.selectbox("Порт посадки", ["S", "C", "Q"], index=0)

    sex_num = 0 if sex == "male" else 1
    embarked_num = {"S": 0, "C": 1, "Q": 2}[embarked]

    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex_num],
        "Age": [age],
        "Fare": [fare],
        "Embarked": [embarked_num]
    })

    if st.button("Предсказать выживаемость"):
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success(f"🎉 Пассажир, вероятно, **выживет** (вероятность {prob:.2%})")
        else:
            st.error(f"💀 Пассажир, вероятно, **не выживет** (вероятность {prob:.2%})")

        st.markdown("#### Введённые данные:")
        st.write(input_data)

# ============================
# TAB 3: О модели
# ============================
with tabs[2]:
    st.subheader("🧠 Информация о модели")

    st.markdown("""
    Модель: **RandomForestClassifier**

    Обучена на признаках:
    - Класс билета (`Pclass`)
    - Пол (`Sex`)
    - Возраст (`Age`)
    - Стоимость билета (`Fare`)
    - Порт посадки (`Embarked`)
    """)

    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": ["Pclass", "Sex", "Age", "Fare", "Embarked"],
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    fig_imp = px.bar(feat_imp, x="Feature", y="Importance", title="Важность признаков")
    st.plotly_chart(fig_imp, use_container_width=True)
