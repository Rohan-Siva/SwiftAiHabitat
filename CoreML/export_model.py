import torch
import torch.nn as nn
import coremltools as ct
import os

class SwiftAIViT(nn.Module):
    def __init__(self, num_classes=5):
        super(SwiftAIViT, self).__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=16, stride=16) 
        self.flatten = nn.Flatten(2) 
        self.transformer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64)
        self.fc = nn.Linear(32 * 14 * 14, num_classes) 

    def forward(self, x):
        # x: [B, 3, 224, 224]
        x = self.conv(x)      # [B, 32, 14, 14]
        x = self.flatten(x)   # [B, 32, 196]
        x = x.permute(2, 0, 1) # [196, B, 32] for Transformer
        x = self.transformer(x)
        x = x.permute(1, 2, 0) # [B, 32, 196]
        x = x.reshape(x.size(0), -1) # [B, 32*196]
        x = self.fc(x)
        return x

labels = ["Graspable", "Sittable", "Movable", "Surface", "Obstacle"]

model = SwiftAIViT(num_classes=len(labels))
model.eval()

example_input = torch.rand(1, 3, 224, 224)

traced_model = torch.jit.trace(model, example_input)

classifier_config = ct.ClassifierConfig(labels)

try:
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="start_image", shape=example_input.shape)], 
        classifier_config=classifier_config,
        minimum_deployment_target=ct.target.iOS17
    )
    
    mlmodel.short_description = "SwiftAI Habitat On-Device ViT"
    mlmodel.author = "Antigravity"
    mlmodel.license = "MIT"
    mlmodel.version = "1.0"
    
    save_path = "SwiftAIHabitatTransformer.mlpackage"
    mlmodel.save(save_path)
    print(f"Model successfully exported to {save_path}")
    
except Exception as e:
    print(f"Error exporting model: {e}")
