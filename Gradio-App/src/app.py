import joblib
import pandas as pd
import numpy as np
import gradio as gr
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import gradio as gr
import joblib
import warnings

warnings.filterwarnings("ignore")

model = joblib.load("models/LR.joblib")

model

test = pd.read_csv("dataframes/Vodafone_churn.csv")
test

# testing our model
model.predict(test)

# creating a function to return a string depending on the output of the model


def classify(num):
    if num == 0:
        return "Customer will not Churn"
    else:
        return "Customer will churn"


"""creating a function for my gradion fn
defining my parameters which my fucntion will accept, and are the same as the features I trained my model on"""


def predict_churn(SeniorCitizen, Partner, Dependents, tenure, InternetService,
                  OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                  StreamingTV, StreamingMovies, Contract, PaperlessBilling,
                  PaymentMethod, MonthlyCharges, TotalCharges):

    # in the code below, I am created a list of my input features

    input_data = [
        SeniorCitizen, Partner, Dependents, tenure, InternetService,
        OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
        StreamingTV, StreamingMovies, Contract, PaperlessBilling,
        PaymentMethod, MonthlyCharges, TotalCharges
    ]
# I am changing my features into a dataframe since that is how I trained my model

    input_df = pd.DataFrame([input_data], columns=[
        "SeniorCitizen", "Partner", "Dependents", "tenure", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges"
    ])

    # I am making a prediction on the input data.
    pred = model.predict(input_df)

    # I am passing the first predction through my classify function I created earlier
    output = classify(pred[0])

    if output == "Customer will not Churn":
        return [(0, output)]
    else:
        # setting my function to return the binary classification and the written output
        return [(1, output)]


output = gr.outputs.HighlightedText(color_map={
    "Customer will not Churn": "green",
    "Customer will churn": "red"
})  # assigning colors to the respective output

# building my interface and wrapping my model in the function

# using gradio blocks to beautify my output

# instatiating my blocks class
block = gr.Blocks()

with block:
    gr.Markdown(""" # Welcome to My Customer Churn Prediction App""")

    input = [gr.inputs.Slider(minimum=0, maximum=1, step=1, label="SeniorCitizen: Select 1 for Yes and 0 for No"),
             gr.inputs.Dropdown(
                 ["Yes", "No"], label="Partner: Do You Have a Partner?"),
             gr.inputs.Dropdown(
                 ["Yes", "No"], label="Dependents: Do You Have a Dependent?"),
             gr.inputs.Number(
                 label="tenure: How Long Have You Been with Vodafone in Months?"),
             gr.inputs.Dropdown(["DSL", "Fiber optic", "No"],
                                label="What Internet Service Do You Use?"),
             gr.inputs.Dropdown(["Yes", "No", "No internet service"],
                                label="Do You Have Online Security?"),
             gr.inputs.Dropdown(["Yes", "No", "No internet service"],
                                label="Do You Have Any Online Backup Service?"),
             gr.inputs.Dropdown(["Yes", "No", "No internet service"],
                                label="Do You Use Any Device Protection?"),
             gr.inputs.Dropdown(["Yes", "No", "No internet service"],
                                label="Do You Use TechSupport?"),
             gr.inputs.Dropdown(["Yes", "No", "No internet service"],
                                label="Do You Stream TV?"),
             gr.inputs.Dropdown(["Yes", "No", "No internet service"],
                                label="Do You Stream Movies?"),
             gr.inputs.Dropdown(["Month-to-month", "One year",
                                 "Two year"], label="What Is Your Contract Type?"),
             gr.inputs.Dropdown(
                 ["Yes", "No"], label=" Do You Use Paperless Billing?"),
             gr.inputs.Dropdown([
                 "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
             ], label="What Payment Method Do You Use?"),
             gr.inputs.Number(label="What is you Monthly Charges?"),
             gr.inputs.Number(label="How Much Is Your Total Charges?")]

    output = gr.outputs.HighlightedText(color_map={
        "Customer will not Churn": "green",
        "Customer will churn": "red"}, label="Your Output")
    predict_btn = gr.Button("Predict")

    predict_btn.click(fn=predict_churn, inputs=input, outputs=output)

block.launch()
