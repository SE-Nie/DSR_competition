from sklearn.metrics import f1_score
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from src.Models import MyModel


def prediction(test_wo_id):
    train_values = pd.read_csv("./data/raw/train_values.csv")
    train_labels = pd.read_csv("./data/raw/train_labels.csv")

    # !!! DROP building_id !!!
    train_values.drop(columns="building_id", inplace=True)
    train_labels.drop(columns="building_id", inplace=True)

    columns_to_target_encode = [
        "geo_level_1_id",
        "geo_level_2_id",
        "geo_level_3_id",
        "ground_floor_type",
    ]
    columns_to_label_encode = [
        "land_surface_condition",
        "foundation_type",
        "roof_type",
        "other_floor_type",
        "position",
        "plan_configuration",
        "legal_ownership_status",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        train_values, train_labels, random_state=42, test_size=0.2
    )

    xg = MyModel(
        model="XGBoost",
        columns_to_labelencode=columns_to_label_encode,
        columns_to_targetencode=columns_to_target_encode,
    )
    xg.fit(X=X_train, y=y_train)
    f1 = xg.get_f1_score(X_test, y_test)
    y_predicted = xg.predict(test_wo_id)
    st.write(f1)
    return y_predicted, f1


st.title("Predictions of 2015 Nepal Earthquake dataset")

st.write("This is a simple example of using Streamlit to create a web app")
st.write("Upload your testdata and get predictions on the building damage grade")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.shape)
    test_wo_id = df.drop(columns="building_id")
    st.dataframe(test_wo_id.head())
    st.button("Predict building damage grade", on_click=prediction(test_wo_id))
