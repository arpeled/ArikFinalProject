
model_path = './ChestX-ray14/pretrained_model.h5'
import torch
import torch.nn as nn
import numpy as np

# דוגמה לארכיטקטורה פשוטה המתאימה למודל שלך
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(128, 64)  # Update to match your architecture
        self.fc2 = nn.Linear(64, 10)   # Update to match your architecture

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# יצירת מודל PyTorch
model = SimpleModel()

# נתיב לקובץ המשקולות (נניח בפורמט NumPy או דומה)
weights_path = "path_to_weights.npy"

# טעינת המשקולות
weights = np.load(weights_path, allow_pickle=True).item()

# מיפוי המשקולות למודל של PyTorch
state_dict = model.state_dict()
for layer_name in weights:
    if layer_name in state_dict:
        state_dict[layer_name] = torch.tensor(weights[layer_name])
    else:
        print(f"Skipping {layer_name} as it is not in the model.")

# טוענים את המשקולות למודל
model.load_state_dict(state_dict)
print("Weights loaded successfully into PyTorch model!")

# שמירת המודל בפורמט PTN
output_path = "converted_model.ptn"
torch.save(model.state_dict(), output_path)
print(f"Model saved at {output_path}")
