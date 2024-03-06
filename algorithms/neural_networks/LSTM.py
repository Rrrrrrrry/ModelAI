import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, input_size=(4, 2), hidden_size=2, num_layers=2):
        print("input_size", input_size)
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size[0], hidden_size=hidden_size, num_layers=num_layers)
        self.lstm2 = nn.LSTM(input_size=input_size[1], hidden_size=hidden_size, num_layers=num_layers)
        self.lr = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        print("x", x.shape)
        output, (hn, cn) = self.lstm1(x)
        output, (hn, cn) = self.lstm2(output)
        output = self.lr(output)[:, -1]
        return output


