import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

# Ensure reproducibility
torch.manual_seed(42)

# Check if MPS (Metal Performance Shaders) is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if device.type == "mps":
    print("PyTorch is using Metal Performance Shaders (MPS) for GPU acceleration.")
else:
    print("PyTorch is not using GPU. Running on CPU.")

# Generate synthetic dataset
input_dim = 1000
output_dim = 10
num_samples = 100000

X = torch.randn(num_samples, input_dim).to(device)
y = torch.randint(0, output_dim, (num_samples,)).to(device)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Define the model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

model = SimpleNN(input_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the model and measure time
epochs = 5

print("Starting PyTorch training...")
start_time = time.time()
for epoch in range(epochs):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
end_time = time.time()

print(f"PyTorch training completed in {end_time - start_time:.2f} seconds")

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X)
    loss = criterion(outputs, y).item()
    accuracy = (outputs.argmax(dim=1) == y).float().mean().item()
print(f"Final Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
