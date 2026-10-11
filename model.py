
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# train data setup
d = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\human activity recognition\train.csv", encoding="latin-1")
trdf = pd.DataFrame(d)
xtrain = trdf.drop(columns=["Activity", "subject"])
ytrain = trdf["Activity"]

# test data setup
d = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\human activity recognition\test.csv", encoding="latin-1")
tedf = pd.DataFrame(d)
x= tedf.drop(columns=["Activity", "subject"])
y = tedf["Activity"]

# test, val split
xval, xtest, yval, ytest = train_test_split(
    x, y, test_size=0.20, random_state=42
)

# target label setup
le = LabelEncoder()
ytrain = le.fit_transform(ytrain)
yval = le.transform(yval)
ytest = le.transform(ytest)

# convert to tensors
xtrain = torch.tensor(xtrain.to_numpy(), dtype=torch.float32)
xtest = torch.tensor(xtest.to_numpy(), dtype=torch.float32)
xval = torch.tensor(xval.to_numpy(), dtype=torch.float32)

ytrain = torch.tensor(ytrain, dtype=torch.long)
ytest = torch.tensor(ytest, dtype=torch.long)
yval = torch.tensor(yval, dtype=torch.long)

# define model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = nn.Linear(561, 128)
        self.relu = nn.ReLU()
        self.m2 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.m1(x)
        x = self.relu(x)
        x = self.m2(x)
        return x

model = MLP()

# loss func
criterion = nn.CrossEntropyLoss()

# optimizer
optim = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 100

# train loop
for e in range(epochs):
    output = model(xtrain)
    loss = criterion(output, ytrain)
    optim.zero_grad()
    loss.backward()
    optim.step()
    with torch.no_grad():
            val_output = model(xval)
            val_loss = criterion(val_output, yval)

with torch.no_grad():
    test_output = model(xtest)
    test_loss = criterion(test_output, ytest)

    predictions = torch.argmax(test_output, dim=1)
    accuracy = (predictions == ytest).float().mean()

print(classification_report(
    ytest.numpy(),
    predictions.numpy(),
    target_names=le.classes_
))

print("Confusion Matrix:")
print(confusion_matrix(ytest.numpy(), predictions.numpy()))

print("Test Loss:", test_loss.item())
print("Test Accuracy:", accuracy.item())
print("val loss", val_loss.item())

# save model
torch.save(model.state_dict(), "model.pth")

# save label encoder
with open("har.pkl", "wb") as f:
    pickle.dump(le, f)
