import pandas as pd
import matplotlib.pyplot as plt

# =============================
# 1. Load dataset
# =============================
df = pd.read_csv("../data/retail_sales.csv")

# Rename columns
df = df.rename(columns={
    "data": "Date",
    "venda": "Sales",
    "estoque": "Stock",
    "preco": "Price"
})

# =============================
# 2. Preprocessing
# =============================
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
df.set_index("Date", inplace=True)

# Keep only sales column
sales_df = df[["Sales"]]

print(sales_df.head())
print(sales_df.info())

# =============================
# 3. Visualization
# =============================
plt.figure(figsize=(10, 5))
plt.plot(sales_df.index, sales_df["Sales"], label="Actual Sales")
plt.title("Historical Retail Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()

# =============================
# 4. Moving Average Forecast
# =============================
sales_df["MA_7"] = sales_df["Sales"].rolling(window=7).mean()

plt.figure(figsize=(10, 5))
plt.plot(sales_df["Sales"], label="Actual Sales")
plt.plot(sales_df["MA_7"], label="7-Day Moving Average")
plt.title("Sales Forecast using Moving Average")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# =============================
# 5. Train-Test Split
# =============================
train_size = int(len(sales_df) * 0.8)
train = sales_df.iloc[:train_size]
test = sales_df.iloc[train_size:]

# =============================
# 6. ARIMA Model
# =============================
model = ARIMA(train["Sales"], order=(5, 1, 0))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))

# =============================
# 7. Evaluation
# =============================
mae = mean_absolute_error(test["Sales"], forecast)
rmse = np.sqrt(mean_squared_error(test["Sales"], forecast))

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# =============================
# 8. Plot Actual vs Forecast
# =============================
plt.figure(figsize=(10, 5))
plt.plot(train.index, train["Sales"], label="Training Data")
plt.plot(test.index, test["Sales"], label="Actual Sales")
plt.plot(test.index, forecast, label="ARIMA Forecast")
plt.title("ARIMA Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()

# =============================
# 9. Save forecast results
# =============================
forecast_df = pd.DataFrame({
    "Date": test.index,
    "Actual_Sales": test["Sales"].values,
    "Predicted_Sales": forecast.values
})

forecast_df.to_csv("../data/forecast_output.csv", index=False)

print("Forecast results saved successfully!")
