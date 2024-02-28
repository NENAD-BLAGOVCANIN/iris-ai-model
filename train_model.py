import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import os


current_dir = os.getcwd()
model_filename = "iris_model.pth"
model_path = os.path.join(current_dir, model_filename)


class Model(nn.Module):

  def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
    super().__init__()
    self.fc1 = nn.Linear(in_features, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)

    return x


torch.manual_seed(41)
model = Model()

#Loading the training data
url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
my_df = pd.read_csv(url)
my_df['variety'] = my_df['variety'].replace('Setosa', 0.0)
my_df['variety'] = my_df['variety'].replace('Versicolor', 1.0)
my_df['variety'] = my_df['variety'].replace('Virginica', 2.0)

my_df

#Defining the input vs output training data
# X is input, y is ouput training data
X = my_df.drop('variety', axis=1)
y = my_df['variety']

X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.002, random_state=41)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#Set the criterion of model to measure the error
criterion = nn.CrossEntropyLoss()
#Choose Adam Optimizer, learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


# Train our model!
# Epochs? (one run thru all the training data in our network)
epochs = 400
losses = []
for i in range(epochs):
  # Go forwared and get a prediction
  y_pred = model.forward(X_train) #get predicted results
  # Mesaure the loss/error, gonna be high at first
  loss = criterion(y_pred, y_train) # predicted value vs the trained value
  losses.append(loss.detach().numpy())

  # print every 10 epoch
  if i%10 == 0:
    print(f'Epohc: {i} and the loss: {loss}')

    # Do some back propagation (fine tune the weghts based on the errors)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Saving model...")

torch.save(model.state_dict(), model_path)

print("Model saved!")