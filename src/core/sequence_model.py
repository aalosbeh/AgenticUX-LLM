"""Optional experimental LSTM sequence model."""

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    optim = None
    TORCH_AVAILABLE = False


class SequenceModelUnavailable(RuntimeError):
    pass


class LSTMRegressor:
    def __init__(self, input_size: int = 13, hidden_size: int = 16, num_layers: int = 1, learning_rate: float = 0.01):
        if not TORCH_AVAILABLE:
            raise SequenceModelUnavailable(
                "PyTorch is not installed. LSTMRegressor is unavailable."
            )
        self.model = _LSTMNet(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, x_seq: np.ndarray, y: np.ndarray, epochs: int = 20) -> None:
        x_tensor = torch.tensor(x_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.model.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            output = self.model(x_tensor)
            loss = self.criterion(output, y_tensor)
            loss.backward()
            self.optimizer.step()

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(x_seq, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_tensor).cpu().numpy().reshape(-1)
        return np.clip(pred, 0, 100)


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        final_state = out[:, -1, :]
        return self.fc(final_state)
