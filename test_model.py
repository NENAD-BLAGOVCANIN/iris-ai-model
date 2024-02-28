import torch
import torch.nn as nn
import torch.nn.functional as F

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


# Instantiate your model (make sure it has the same architecture as the saved model)
model = Model()

# Load the saved parameters from the .pth file
model.load_state_dict(torch.load("iris_model.pth"))

# Set the model to evaluation mode
model.eval()

# Define your input data (example)
input_data = torch.FloatTensor([6.1, 3, 4.9, 1.8])

# Pass the input data through the model to obtain predictions
with torch.no_grad():
    output = model(input_data)

# Process the output as needed (e.g., get the predicted class for classification)
predicted_class = torch.argmax(output).item()

print("Predicted class:", predicted_class)
