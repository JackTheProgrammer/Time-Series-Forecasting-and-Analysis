import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
# from sklearn.model_selection import GridSearchCV
import numpy as np
import os

# Load or fetch data
def load_data():
    if os.path.exists("sp500.csv"):
        sp500 = pd.read_csv("sp500.csv", index_col=0)
    else:
        sp500 = yf.Ticker("^GSPC")
        sp500 = sp500.history(period="max")
        sp500.to_csv("sp500.csv")
    sp500.index = pd.to_datetime(sp500.index)
    return sp500

# Preprocess data
def preprocess_data(sp500):
    sp500 = sp500.copy()
    sp500.drop(columns=["Dividends", "Stock Splits"], inplace=True)
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    sp500 = sp500.loc["1990-01-01":].copy()
    return sp500

# Feature engineering
def add_features(sp500):
    horizons = [2, 5, 60, 250, 1000]
    for horizon in horizons:
        rolling_avg = sp500["Close"].rolling(horizon).mean()
        sp500[f"Close_Ratio_{horizon}"] = sp500["Close"] / rolling_avg
        sp500[f"Trend_{horizon}"] = sp500["Target"].shift(1).rolling(horizon).sum()
    sp500["Volatility_5"] = sp500["Close"].rolling(5).std()
    sp500["Volatility_20"] = sp500["Close"].rolling(20).std()
    sp500.dropna(inplace=True)
    return sp500

# Backtesting framework
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    probs = model.predict_proba(test[predictors])[:, 1]
    preds = (probs >= 0.6).astype(int)
    predictions = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], predictions], axis=1)
    return combined

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

# Enhanced evaluation
def evaluate(predictions):
    precision = precision_score(predictions["Target"], predictions["Predictions"], zero_division=0)
    recall = recall_score(predictions["Target"], predictions["Predictions"], zero_division=0)
    f1 = f1_score(predictions["Target"], predictions["Predictions"], zero_division=0)
    accuracy = np.mean(predictions["Target"] == predictions["Predictions"])
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    return precision, recall, f1, accuracy

# Main workflow
if __name__ == "__main__":
    # Load and preprocess data
    sp500 = load_data()
    sp500 = preprocess_data(sp500)
    sp500 = add_features(sp500)

    # Define model and predictors
    predictors = ["Close", "Volume", "Open", "High", "Low"] + \
                 [f"Close_Ratio_{h}" for h in [2, 5, 60, 250, 1000]] + \
                 [f"Trend_{h}" for h in [2, 5, 60, 250, 1000]] + \
                 ["Volatility_5", "Volatility_20"]

    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

    # Backtest and evaluate
    predictions = backtest(sp500, model, predictors)
    evaluate(predictions)