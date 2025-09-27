import torch
import torch.nn as nn

class LSTMIDS(nn.Module):
    def __init__(self, input_size=25, hidden_size=64, num_layers=1, dropout=0.2):
        super(LSTMIDS, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence length dimension
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.sigmoid(out)