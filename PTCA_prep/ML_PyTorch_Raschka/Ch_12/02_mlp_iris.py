import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

iris = load_iris()
X = iris["data"]
y = iris["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1.0 / 3, random_state=42
)

X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
X_train_norm = torch.from_numpy(X_train_norm).float()
y_train = torch.from_numpy(y_train)

torch.manual_seed(1)

train_ds = TensorDataset(X_train_norm, y_train)
train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)


class FlowerModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, X):
        X = self.layer1(X)
        X = torch.nn.Sigmoid()(X)
        X = self.layer2(X)

        return X


input_size = X_train_norm.shape[1]
hidden_size = 16
output_size = 3

flower_model = FlowerModel(input_size, hidden_size, output_size)
lr = 0.0005
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(flower_model.parameters(), lr=lr)

num_epochs = 1000
loss_hist = [0] * num_epochs
acc_hist = [0] * num_epochs

for epoch in range(num_epochs):
    for X_batch, y_batch in train_dl:
        pred = flower_model(X_batch)
        loss = loss_fn(pred, y_batch)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_hist[epoch] = loss.item() * y_batch.size(0)
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        acc_hist[epoch] += is_correct.sum()

    loss_hist[epoch] /= len(train_dl.dataset)
    acc_hist[epoch] /= len(train_dl.dataset)


fig = plt.figure(figsize=(12, 5))

ax = fig.add_subplot(1, 2, 1)
ax.plot(loss_hist, lw=3)
ax.set_title("Training loss", size=15)
ax.set_xlabel("Epoch", size=15)
ax.tick_params(axis="both", which="major", labelsize=15)

ax = fig.add_subplot(1, 2, 2)
ax.plot(acc_hist, lw=3)
ax.set_title("Training accuracy", size=15)
ax.set_xlabel("Epoch", size=15)
ax.tick_params(axis="both", which="major", labelsize=15)
plt.show()


# ----------------------------------- Test ----------------------------------- #
print(
    f"\n\n\n# ----------------------------------- #\n\nTest\n\n# ----------------------------------- #\n\n\n"
)

X_test_norm = (X_test - np.mean(X_train)) / (np.std(X_train))
X_test_norm = torch.from_numpy(X_test_norm).float()
y_test = torch.from_numpy(y_test)

pred_test = flower_model(X_test_norm)
correct = (torch.argmax(pred_test, dim=1) == y_test).float()
test_acc = correct.mean()
print(f"Test accuracy: {test_acc:.4f}")
