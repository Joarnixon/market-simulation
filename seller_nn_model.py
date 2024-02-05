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

class SellerCNN(nn.Module):
    def __init__(self, N: int):
        super().__init__()
        # We have 1xNx3 -> 3x(N-1)x2 -> 3x(N-2)x1 -> 3x1x1 -> 3x1 vector.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size = (2, 2), padding=0, dtype=torch.float)
        self.activation1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size = (2, 2), padding=0, dtype=torch.float)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=N-2, out_features=1, dtype=torch.float)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        return x

    def predict(self, x):
        x = array(x)
        x = torch.tensor(x, dtype=torch.float)
        x = x.unsqueeze(0)
        with torch.inference_mode():
            answer = self.forward(x)
        return answer


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


def train_generative(x, y, model_generative, model_head, epochs):
    model_generative.train()
    optimizer = torch.optim.Adam(params=model_generative.parameters(), lr=3e-3)
    x = array(x)
    y = array(y)
    x = torch.tensor(x, dtype=torch.float)
    total_loss = 0
    for i in range(epochs):
        predictions = model_generative(x)
        head_prediction = model_head(predictions)
        if head_prediction > 0:
            error_scale = 1
        else:
            error_scale = -4
        # Thus we ensure that loss is greater for values close to 0 (we want the model to actually make profit) and less for bigger values.
        loss = (head_prediction + 1/head_prediction ) * error_scale
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss

def train_epochs(x, y, model, epochs):
    torch.save(model.state_dict(), 'model.pth')
    torch.save(x, 'x_data.pth')
    torch.save(y, 'y_data.pth')
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

model = SellerNN()
model.load_state_dict(torch.load('model.pth'))
x = torch.load('x_data.pth')
y = torch.load('y_data.pth')
print(x, y, model)

