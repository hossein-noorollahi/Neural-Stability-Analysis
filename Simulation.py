"""
Neural Network Stability Analysis under Synaptic Perturbation
Description: This script analyzes the robustness of a trained MLP against 
structural weight perturbations (Gaussian noise) and measures the metabolic 
cost (epochs) required for functional recovery.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import copy
import os

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Data Path (Creates a local 'data' folder)
current_dir = os.getcwd()
data_path = os.path.join(current_dir, 'data')

if not os.path.exists(data_path):
    os.makedirs(data_path)

print(f"🚀 Initializing Simulation on {device}...")

# ==========================================
# 2. MODEL ARCHITECTURE (Standard MLP)
# ==========================================
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Data Loading
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

print("📂 Loading MNIST Dataset...")
try:
    train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
except RuntimeError:
    print("⚠️ Download fallback: Checking for local files...")
    train_dataset = datasets.MNIST(root=data_path, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=data_path, train=False, download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

model = SimpleNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

def evaluate(model_instance):
    model_instance.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model_instance(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(test_loader.dataset)

# ==========================================
# 3. PHASE 1: INITIAL TRAINING (Baseline)
# ==========================================
print("\n🧠 Phase 1: Establishing Baseline Performance...")
for epoch in range(1, 6):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
    
    acc = evaluate(model)
    print(f'Epoch {epoch}: Accuracy = {acc:.2f}%')

original_model = copy.deepcopy(model)
original_acc = evaluate(original_model)

# ==========================================
# 4. PHASE 2: SENSITIVITY ANALYSIS (Perturbation)
# ==========================================
print("\n⚡ Phase 2: Analyzing Sensitivity to Structural Noise...")
perturbation_levels = np.linspace(0, 0.5, 20)
accuracies = []

for p in perturbation_levels:
    perturbed_model = copy.deepcopy(original_model)
    with torch.no_grad():
        for param in perturbed_model.parameters():
            noise = torch.randn_like(param) * p
            param.add_(noise)
    accuracies.append(evaluate(perturbed_model))

# Plot Figure 1
plt.figure(figsize=(10, 6))
plt.plot(perturbation_levels, accuracies, marker='o', color='crimson', linewidth=3)
plt.title(f'Network Stability under Weight Perturbation\n(Base Accuracy: {original_acc:.1f}%)', fontsize=14)
plt.xlabel('Noise Standard Deviation (σ)', fontsize=12)
plt.ylabel('Inference Accuracy (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('Sensitivity_Analysis.png', dpi=300)
print("✅ Sensitivity Graph saved as 'Sensitivity_Analysis.png'")

# ==========================================
# 5. PHASE 3: RECOVERY COST ANALYSIS
# ==========================================
print("\n🔋 Phase 3: Measuring Recovery Cost (Retraining)...")

baseline_accs = [original_acc] * 16
perturbed_model = copy.deepcopy(original_model)
optimizer_repair = optim.SGD(perturbed_model.parameters(), lr=0.01, momentum=0.5)

# Inject Noise (σ=0.5)
with torch.no_grad():
    for param in perturbed_model.parameters():
        noise = torch.randn_like(param) * 0.5 
        param.add_(noise)

recovery_accs = [evaluate(perturbed_model)]
print(f"Accuracy after Perturbation: {recovery_accs[0]:.2f}%")

for epoch in range(1, 16):
    perturbed_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer_repair.zero_grad()
        loss = criterion(perturbed_model(data), target)
        loss.backward()
        optimizer_repair.step()
    recovery_accs.append(evaluate(perturbed_model))
    print(f"Recovery Epoch {epoch}: {recovery_accs[-1]:.2f}%")

# Plot Figure 2
epochs_range = range(0, 16)
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, baseline_accs, label='Baseline (Unperturbed)', color='green', linestyle='--', linewidth=2)
plt.plot(epochs_range, recovery_accs, label='Retraining Trajectory', color='blue', marker='o', linewidth=2)
plt.fill_between(epochs_range, recovery_accs, baseline_accs, color='gray', alpha=0.2, label='Computational Deficit')
plt.title('Computational Cost of Recovery', fontsize=14)
plt.xlabel('Training Epochs', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.5)
plt.savefig('Recovery_Cost.png', dpi=300)
print("✅ Recovery Graph saved as 'Recovery_Cost.png'")

print("\n🎉 Analysis Complete.")