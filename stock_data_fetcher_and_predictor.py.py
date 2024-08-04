import requests
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Function to fetch stock data from Alpha Vantage
def fetch_stock_data(api_key, symbol):
    print("Fetching stock data...")
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if "Time Series (Daily)" not in data:
        raise ValueError("Error fetching data from Alpha Vantage. Please check your API key and symbol.")
    
    time_series = data["Time Series (Daily)"]
    
    df = pd.DataFrame.from_dict(time_series, orient="index", dtype=float)
    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. volume": "volume"
    })
    df.index = pd.to_datetime(df.index)
    
    print(f"Data shape: {df.shape}")
    print(df.head())
    
    return df

api_key = " DCAL4944133ENHGP"  # Replace with your actual API key
symbol = "AC.TO"  # ARI Canada ticker
stock_data = fetch_stock_data(api_key, symbol)

# Data Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(stock_data[['open', 'high', 'low', 'close', 'volume']])
stock_tensor = torch.tensor(scaled_features, dtype=torch.float32)

# Set targets
target_scaler = StandardScaler()
target_values = stock_data['close'].values.reshape(-1, 1)
target_scaler.fit(target_values)
target = torch.tensor(target_scaler.transform(target_values), dtype=torch.float32)

# Define a simple linear regression model
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

input_dim = stock_tensor.shape[1]
output_dim = 1

model = SimpleModel(input_dim, output_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting training loop...")

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(stock_tensor)
    loss = criterion(outputs, target)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

print("Training complete. Generating predictions...")

model.eval()
with torch.no_grad():
    predictions = model(stock_tensor)
    predictions = target_scaler.inverse_transform(predictions.numpy())  # Convert back to original scale

# Convert predictions tensor to DataFrame
predictions_df = pd.DataFrame(predictions, columns=['Predicted Close'])
stock_data.reset_index(inplace=True)
result_df = pd.concat([stock_data, predictions_df], axis=1)

# Save DataFrame to CSV
csv_filename = 'stock_predictions.csv'
result_df.to_csv(csv_filename, index=False)
print(f"Predictions saved to {csv_filename}")

print(result_df.head())
