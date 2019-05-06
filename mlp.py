import torch
import torch.nn as nn

# Fully connected neural network with one hidden layer


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class MLP:
    def __init__(self, lr=0.001, input_size=10, hidden_size=10, output_dim=1, epochs=5, batch_size=1):
        self.epochs = epochs
        self.device = torch.device('cpu')
        self.batch_size = batch_size
        self.model = NeuralNet(input_size=input_size, hidden_size=hidden_size,
                               output_dim=output_dim).to(self.device).double()
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        print(self.device)

    def fit(self, x_train, y_train):
        self.model.fc1.reset_parameters()
        self.model.fc2.reset_parameters()
        # Train the model
        total_step = len(x_train)
        for epoch in range(self.epochs):
            for i, (x, y) in enumerate(zip(x_train, y_train)):
                # Move tensors to the configured device
                x = torch.tensor(x).to(self.device)
                y = torch.tensor(y).view(1).double().to(self.device)

                # Forward pass
                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, x_test):
        # Test the model
        # In test phase, we don't need to compute gradients (for memory efficiency)
        with torch.no_grad():
            x = torch.tensor(x_test).to(self.device)
            outputs = self.model(x).cpu()
        return outputs.numpy()


if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    X, y = make_regression(n_samples=2000, n_features=10, noise=2, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    model = MLP()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print(pred, pred.shape)