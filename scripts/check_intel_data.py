import pandas as pd

# Read price history
print("Price History (first 5 rows):")
df = pd.read_csv("intc_daily_prices.csv", index_col=0)
print(df.head())
print("\nDataset Info:")
print(df.info())

print("\nFundamental Data:")
fund = pd.read_csv("intc_fundamentals.csv")
print(fund.T)  # Transpose for better viewing