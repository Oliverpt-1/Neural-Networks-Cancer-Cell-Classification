
#pytorch neural network
    #Details
        #4 Hidden Layers with ReLu activations
        #Dropout layers are added after each hidden layer to prevent overfitting 
        #Sigmoid activation for output layer - binary classification
        #Uses ADAM optimization rather than Stochastic Gradient Descent(gives good results for Hyperparameter tuning)
         #Loss Function - Binary cross Entropy


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Import our data
column_names = [
    "ID", "Diagnosis", 
    "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness",
    "Mean Compactness", "Mean Concavity", "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
    "Radius SE", "Texture SE", "Perimeter SE", "Area SE", "Smoothness SE",
    "Compactness SE", "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE",
    "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area", "Worst Smoothness",
    "Worst Compactness", "Worst Concavity", "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"
]

data = pd.read_csv('data/wdbc.data')
n_feat = 30
data.columns = column_names

data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})  # Convert M to 1, B to 0

df_numeric = data.drop(columns=['ID'], errors='ignore')
x = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

total_samples = len(x)
train_size = int(total_samples * 0.8)
val_size = int(total_samples * 0.1)

X_train = x[:train_size]
y_train = y[:train_size]
X_val = x[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]

X_test = x[train_size+val_size:]
y_test = y[train_size+val_size:]

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val).unsqueeze(1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

class EnhancedNN(nn.Module):  # inherit from nn Module
    def __init__(self, dropout_rate=0.5):
        # Call the __init__ method of the base class nn.Module
        super(EnhancedNN, self).__init__()

        # Define the network structure using nn.Sequential
        # This stacks layers together in the order they are written
        self.net = nn.Sequential(
            nn.Linear(n_feat, 50),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(50, 40),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(40, 30),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    # Forward method defines how data passes through the model
    def forward(self, x):
        # Pass input x through the defined object "net" (the layers above)
        return self.net(x)

model = EnhancedNN(dropout_rate=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # L2 regularization with weight decay
loss_function = nn.BCELoss()

loss_train = []
loss_val = []

for epoch in range(500):
    # Zero out gradients for this pass
    optimizer.zero_grad()
    
    # Predict on train data
    output = model(X_train)
    
    # Compute train loss
    loss = loss_function(output, y_train)
    loss_train.append(loss.item())
    
    # Predict and compute loss on validation data
    with torch.no_grad():  # No need to compute gradients for validation data
        out_val = model(X_val)
        val_loss = loss_function(out_val, y_val)
        loss_val.append(val_loss.item())  # Append the scalar value of the loss
    
    # Backpropagation, auto-differentiation
    loss.backward()
    
    # Update weights
    optimizer.step()

# Plot loss curves
plt.plot(loss_train, label="Train Loss")
plt.plot(loss_val, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Print the learned parameters
print("Learned Parameters (weights and biases):")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")

# Test set evaluation
with torch.no_grad():
    # Get model predictions
    preds = model(X_test)
    
    # Convert probabilities to binary predictions (0 or 1)
    binary_preds = (preds > 0.5).float()
    
    # Calculate accuracy
    accuracy = (binary_preds == y_test).float().mean()
    print(f"Test Accuracy: {accuracy.item():.4f}")
