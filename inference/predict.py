import torch
import joblib
import numpy as np
import sys

from model_config import MODEL_INPUT_DIM
sys.path.append('..')
from model.PredictorModel import HousePricePredictor

# 1. Load the Scalers
scaler_x = joblib.load("scaler_x.joblib")
scaler_y = joblib.load("scaler_y.joblib")

# 2. Load and Prepare the Model

model = HousePricePredictor(MODEL_INPUT_DIM)
model.load_state_dict(torch.load("model_parameters.pth"))
model.eval() # Set to evaluation mode

def predict(raw_input_data):
    """
    raw_input_data: list or numpy array of raw features
    """
    with torch.no_grad(): # Disable gradient calculation for speed/memory
        # A. Scale the Input (X)
        # Reshape to (1, -1) if it's a single sample
        input_array = np.array(raw_input_data).reshape(1, -1)
        scaled_input = scaler_x.transform(input_array)

        # B. Convert to Tensor
        input_tensor = torch.FloatTensor(scaled_input)

        # C. Forward Pass
        scaled_prediction = model(input_tensor)

        # D. Inverse Scale the Output (Y)
        prediction_np = scaled_prediction.numpy()
        final_prediction = scaler_y.inverse_transform(prediction_np)

        return final_prediction[0][0]

if __name__ == "__main__":
    # Example usage:
    new_data = [-122.22,37.86,21.0,7099.0,1106.0,2401.0,1138.0,8.3014] # Your raw feature values
    result = predict(new_data)
    print(f"Prediction in original units: {result}")