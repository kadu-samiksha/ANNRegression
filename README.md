# Estimated Salary Prediction (ANN Regression)

A regression project that predicts a bank customer's estimated salary from their profile, using an Artificial Neural Network built with TensorFlow/Keras and served through a Streamlit web app.

## Problem Statement

Given a customer's attributes (credit score, geography, age, balance, whether they exited, etc.), predict their `EstimatedSalary` (a continuous value).

## Dataset

- **File:** `Churn_Modelling.csv` (10,000 rows)
- **Target:** `EstimatedSalary`
- **Features used:** `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `Exited`
- **Dropped:** `RowNumber`, `CustomerId`, `Surname`

## Project Structure

```
salary-regression/
├── Churn_Modelling.csv          # Dataset
├── salaryRegression.ipynb       # Preprocessing, ANN regression training, evaluation
├── app2.py                      # Streamlit app
├── regression_model.h5          # Trained ANN regression model
├── label_encoder_gender.pkl     # Fitted LabelEncoder (Gender)
├── onehot_encoder_geo.pkl       # Fitted OneHotEncoder (Geography)
├── scaler.pkl                   # Fitted StandardScaler
├── requirements.txt
└── README.md
```

## Workflow

1. **Preprocessing** (`salaryRegression.ipynb`)
   - Drop identifier columns
   - Label-encode `Gender`
   - One-hot encode `Geography` (`handle_unknown='ignore'`)
   - 80/20 train-test split (`random_state=42`)
   - Standardize features with `StandardScaler` (fit on train only)
   - Save the encoders and scaler as `.pkl` files
2. **Model**
   - Input layer: 12 features
   - Dense(64, ReLU) → Dense(32, ReLU) → Dense(1) (linear output)
   - Optimizer: Adam
   - Loss: Mean Absolute Error, Metric: MAE
   - Callbacks: EarlyStopping (`val_loss`, patience = 10, restore best weights), TensorBoard
3. **Inference** (`app2.py`)
   - Apply the saved encoders and scaler to new input and output the predicted salary

## Results

| Metric | Value |
|--------|-------|
| Test MAE | _fill in from your notebook_ |

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

**Run the web app**

```bash
streamlit run app2.py
```

**Retrain the model**

Open `salaryRegression.ipynb` and run all cells. This regenerates `regression_model.h5`, `scaler.pkl`, and both encoder `.pkl` files.

**View training curves**

```bash
tensorboard --logdir regressionlogs/fit
```

## Tech Stack

Python · TensorFlow / Keras · scikit-learn · pandas · NumPy · Streamlit · TensorBoard

## Notes

- The scaler here is fit on a different feature set (includes `Exited`, excludes `EstimatedSalary`) than in the classification project. Do not share `scaler.pkl` between the two projects.
- Feature column order at inference must match training order.
- `EstimatedSalary` in this dataset is synthetic and weakly correlated with the other features, so expect a modest MAE. The project is mainly a demonstration of the ANN regression workflow.

## Future Improvements

- Add R² and RMSE alongside MAE
- Tune hyperparameters (neurons, layers, epochs) with a proper regression setup
- Scale the target variable
- Compare against linear regression and tree-based baselines
- Deploy on Streamlit Community Cloud
