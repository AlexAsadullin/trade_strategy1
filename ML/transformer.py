import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from custom_datasets import TimeSeriesDataset
from custom_metrics import directional_accuracy_score
from data_manipulations import split_data, prepare_data_ratio


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=5, dim_feedforward=256, device='cpu', loss_function=nn.MSELoss()):
        super().__init__()
        self.device = device
        self.loss_function = loss_function
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True) # batch_first=False
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.to(device)
    
    def forward(self, x):
        x = self.embedding(x) #.unsqueeze(1)  # Ensure (batch_size, 1, d_model)
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :]).squeeze()
    
    def train_validate_one_epoch(self, train_loader, optimizer, epoch):
        self.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            optimizer.zero_grad()
            output = self.forward(x_batch)
            #output = output.view(-1, 1)
            loss = self.loss_function(output, y_batch.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")
        return total_loss / len(train_loader)

def main(data_read_path: str, model_save_path: str, train_part: float = 0.8):
    num_epochs = 30
    scaler = MinMaxScaler()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = pd.read_csv(data_read_path, index_col=0)
    try:
        df = df.drop('Time', axis='columns')
    except Exception as e: print(e)

    df = prepare_data_ratio(df=df, n_ratio=10, window_size=40)
    X_train, X_test, y_train, y_test = split_data(df, train_part, scaler)

    # sequence_length = 20
    
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    input_dim = X_train.shape[1]
    model = TransformerModel(input_dim, device=device, loss_function=nn.L1Loss())
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # train model
    for epoch in range(num_epochs):
        avg_loss = model.train_validate_one_epoch(train_loader, optimizer, epoch)

    torch.save(model.state_dict(), model_save_path)

    model.eval()
    actuals, predictions = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_pred = model(x_batch).cpu().numpy()
            predictions.extend(y_pred)
            actuals.extend(y_batch.numpy())

    print(directional_accuracy_score(actuals=np.array(actuals), predictions=np.array(predictions)))

if __name__ == "__main__":
    main("/home/alex/BitcoinScalper/data_collecting/tinkoff_data/prices_massive_LKOH_4_HOUR_2025-01-25.csv",
         "/home/alex/BitcoinScalper/ML/models/transformer_model.pkl")