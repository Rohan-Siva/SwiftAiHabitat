import torch
import torch.nn as nn
import coremltools as ct

# Define a simple Vision Transformer-like model
class SimpleViT(nn.Module):
    def __init__(self):
        super(SimpleViT, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 111 * 111, 10) # Dummy output size for 224x224 input

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Initialize model
model = SimpleViT()
model.eval()

# Create dummy input
example_input = torch.rand(1, 3, 224, 224)

# Trace the model
traced_model = torch.jit.trace(model, example_input)

# Convert to Core ML
try:
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="image", shape=example_input.shape)],
        minimum_deployment_target=ct.target.iOS17
    )
    
    mlmodel.save("SwiftAIHabitatTransformer.mlpackage")
    print("Model successfully exported to SwiftAIHabitatTransformer.mlpackage")
except Exception as e:
    print(f"Error exporting model: {e}")
