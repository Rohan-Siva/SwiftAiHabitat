# SwiftAI Habitat

**On-Device AI Environment Simulator**

SwiftAI Habitat is an iOS application that leverages **LiDAR** and **ARKit** to scan real-world rooms, then uses **Core ML** with a custom **Vision Transformer (ViT)** model to simulate dynamic environmental changes. It showcases Apple Intelligence's on-device Foundation Models for semantic understanding fused with the Vision framework for real-time scene analysis.

## Features

- **Real-time Room Scanning**: Uses ARKit and LiDAR to capture 3D meshes and depth data.
- **Object Understanding**: Integrates Vision framework for object saliency and segmentation.
- **On-Device Inference**: Runs a custom Transformer model via Core ML to predict object affordances and interactions.
- **Interactive "What-If" Scenarios**: Allows users to rearrange furniture and visualize robot path planning or design changes in AR.
- **Privacy First**: All processing happens on-device using the Neural Engine.

## Architecture

- **MVVM Pattern**: Clean separation of logic and UI.
- **RealityKit**: High-performance AR rendering.
- **Core ML**: Custom model integration.
- **SwiftUI**: Modern, declarative user interface.

## Requirements

- iOS 17.0+
- Device with LiDAR Scanner (iPhone 12 Pro or later, iPad Pro)
- Xcode 15.0+

## Setup

1. Open the project in Xcode.
2. Ensure you have a valid signing team selected.
3. Build and run on a physical device (Simulator does not support Camera/LiDAR).

## License

MIT
