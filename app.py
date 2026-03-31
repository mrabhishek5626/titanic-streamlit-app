
# import packages
import streamlit as st

import pandas as pd

import joblib

from joblib import load

# load the dependencies

model = load('tree.joblib')

train_cols = load('columns.joblib')

data = load('data.joblib')

# page configurations

st.set_page_config(page_title="First User Interface", layout="wide")

# title

st.title("🚢 Titanic Streamlit")

st.markdown("## *Interactive ML Dashboard with Titanic UI*")

st.divider()

# sidebar panel

st.sidebar.header("🎯 Passenger Details")

Pclass = st.sidebar.radio("Passenger Class", [1, 2, 3])

Sex = st.sidebar.radio("Sex", ["Male", "Female"])
Sex = 1 if Sex == "Male" else 0

Embarked = st.sidebar.selectbox("Embarked", ["Cherbourg", "Queenstown", "Southampton"])
emb_map = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}
Embarked = emb_map[Embarked]

Age_cat = st.sidebar.slider("Age Category (0:Child ---- 3:Senior)", 0, 3, 1)

Fare_cat = st.sidebar.slider("Fare Category (0:Low ---- 3:High)", 0, 3, 1)

Family = st.sidebar.radio("Family Onboard?", ["No", "Yes"])
Family = 1 if Family == "Yes" else 0

# input dataframe

input_df = pd.DataFrame({
    
    'Pclass': [Pclass],
    
    'Sex': [Sex],
    
    'Embarked': [Embarked],
    
    'Age Category': [Age_cat],
    
    'Fare Category': [Fare_cat],
    
    'Family': [Family]
})

input_df = input_df.reindex(columns = train_cols, fill_value = 0)


# data insights

st.subheader("📊 Dataset Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Passengers", len(data))
    st.bar_chart(data['Pclass'].value_counts())

with col2:
    male_count = int((data['Sex'] == 1).sum())
    female_count = int((data['Sex'] == 0).sum())

    m1, m2 = st.columns(2)

    with m1:
        st.metric("Female", female_count)

    with m2:
        st.metric("Male", male_count)

    st.bar_chart(data['Sex'].value_counts())

with col3:
    st.metric("Unique Embarked", data['Embarked'].nunique())
    st.bar_chart(data['Embarked'].value_counts())

st.divider()


# predictions

st.subheader("🔮 Prediction Panel")

col_pred, col_prob = st.columns(2)

with col_pred:
    if st.button("Predict"):
        result = model.predict(input_df)
        prob = model.predict_proba(input_df)

        if result[0] == 1:
            st.success("Passenger Survived ✅")
        else:
            st.error("Passenger Did Not Survive ❌")

        st.session_state['prob'] = prob

with col_prob:
    if 'prob' in st.session_state:
        st.metric("Survival Probability", f"{round(st.session_state['prob'][0][1]*100,2)}%")

# reset button

if st.button("Reset"):
    st.session_state.clear()
    st.rerun()

st.divider()

# input button 

st.subheader("🧾 Input Summary")
st.dataframe(input_df)