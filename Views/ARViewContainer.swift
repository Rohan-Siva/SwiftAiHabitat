import SwiftUI
import RealityKit

struct ARViewContainer: UIViewRepresentable {
    @EnvironmentObject var arViewModel: ARViewModel
    
    func makeUIView(context: Context) -> ARView {
        return arViewModel.arService.arView
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {}
}
