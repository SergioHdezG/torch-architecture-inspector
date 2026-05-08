import torch
import torch.nn as nn
from torch_arch_inspector import generate_model_inspector

# 1. Example with a Generic Model (No extra dependencies)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 112 * 112, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))

print("Generating report for SimpleCNN...")
generic_model = SimpleCNN()
generate_model_inspector(generic_model, output_file="generic_model_report.html")

# 2. Example with AlphaGenome (Optional/External dependency)
try:
    from alphagenome_pytorch import AlphaGenome
    print("\nGenerating report for AlphaGenome...")
    genome_model = AlphaGenome(num_organisms=2)
    genome_model.eval()
    generate_model_inspector(genome_model, output_file="alphagenome_report.html")
except ImportError:
    print("\nAlphaGenome not found, skipping second example.")
