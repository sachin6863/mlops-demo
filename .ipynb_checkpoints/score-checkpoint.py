import argparse
import pandas as pd
import joblib
import os
from azureml.core import Run
from azureml.core.model import Model

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
parser.add_argument("--output_data", type=str)
parser.add_argument("--model_path", type=str)

args = parser.parse_args()

# Load input
df = pd.read_csv(args.input_data)

# Load model
model = joblib.load(os.path.join(args.model_path, "model.pkl"))

# Predict
df["prediction"] = model.predict(df)

# Save output
os.makedirs(args.output_data, exist_ok=True)
df.to_csv(os.path.join(args.output_data, "predictions.csv"), index=False)
