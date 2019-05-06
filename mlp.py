import torch
import torch.nn as nn

# Fully connected neural network with one hidden layer


class MLP(nn.Module):
    def __init__(self, input_size=10, hidden_size=10, output_dim=1, lr=0.001, epochs=5, batch_size=1):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.criterion = nn.MSELoss()
        self.model = MLP()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def model(self):
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def fit(self, x_train, y_train):
        # Train the model
        total_step = len(x_train)
        for epoch in range(self.epochs):
            for i, (x, y) in enumerate(zip(x_train, y_train)):
                # Move tensors to the configured device
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass
                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, self.epochs, i + 1, total_step, loss.item()))

    def predict(self, x_test):
        # Test the model
        # In test phase, we don't need to compute gradients (for memory efficiency)
        with torch.no_grad():
            for x in x_test:
                x = x.to(self.device)
                outputs = self.model(x)
                return outputs


if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    X, y = make_regression(n_samples=1000, n_features=10, noise=5, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    model = MLP()
    model.fit(x_train, y_train)
    model.predict(x_test)