import torch
from torch import nn
from numpy import array


class SellerNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 3, dtype=torch.float)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(3, 3, dtype=torch.float)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(3, 3, dtype=torch.float)
        self.activation3 = nn.ReLU()
        self.linear4 = nn.Linear(3, 3, dtype=torch.float)
        self.activation4 = nn.ReLU()
        self.linear5 = nn.Linear(3, 1, dtype=torch.float)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.activation3(x)
        x = self.linear4(x)
        x = self.activation4(x)
        x = self.linear5(x)
        return x

    def predict(self, x):
        x = array(x)
        x = torch.tensor(x, dtype=torch.float)
        x = x.unsqueeze(0)
        with torch.inference_mode():
            answer = self.forward(x)
        return answer

    def get_grad(self, x):
        x = array(x)
        x = torch.tensor(x, dtype=torch.float, requires_grad=True)
        output = self.forward(x)
        output.backward()
        return x.grad


class CustomLoss(nn.Module):
    def __init__(self, weight_positive=1, weight_negative=2):
        super().__init__()
        self.weight_positive = weight_positive
        self.weight_negative = weight_negative

    def forward(self, predicted, target):
        loss = (predicted - target)**2
        positive_mask = target > 0
        negative_mask = target <= 0

        loss[positive_mask] *= self.weight_positive
        loss[negative_mask] *= self.weight_negative

        return torch.sum(loss)


def train_epochs(x, y, model, epochs):
    model.train()
    loss_function = CustomLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-3)
    x = array(x)
    y = array(y)
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    y = y.unsqueeze(1)
    total_loss = 0
    for i in range(epochs):
        predictions = model(x)
        loss = loss_function(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss

