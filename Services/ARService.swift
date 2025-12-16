import ARKit
import RealityKit
import Combine

class ARService: NSObject, ObservableObject, ARSessionDelegate {
    var arView: ARView
    
    @Published var currentFrame: ARFrame?
    
    override init() {
        arView = ARView(frame: .zero)
        super.init()
        
        setupAR()
    }
    
    func setupAR() {
        let config = ARWorldTrackingConfiguration()
        
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
        }
        
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        
        config.planeDetection = [.horizontal, .vertical]
        
        arView.session.delegate = self
        arView.session.run(config)
    }
    
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        DispatchQueue.main.async {
            self.currentFrame = frame
        }
    }
    
    func addVirtualObject(at position: SIMD3<Float>) {
        let anchor = AnchorEntity(world: position)
        // Placeholder mesh
        let mesh = MeshResource.generateBox(size: 0.1)
        let material = SimpleMaterial(color: .blue, isMetallic: true)
        let entity = ModelEntity(mesh: mesh, materials: [material])
        
        anchor.addChild(entity)
        arView.scene.addAnchor(anchor)
    }
    
    // MARK: - Interaction & Debugging
    
    func raycast(from point: CGPoint) -> ARRaycastResult? {
        let query = arView.makeRaycastQuery(from: point, allowing: .estimatedPlane, alignment: .any)
        guard let raycastQuery = query else { return nil }
        
        let results = arView.session.raycast(raycastQuery)
        return results.first
    }
    
    func toggleMeshVisualization(_ enabled: Bool) {
        if enabled {
            arView.debugOptions.insert(.showSceneUnderstanding)
        } else {
            arView.debugOptions.remove(.showSceneUnderstanding)
        }
    }
    
    func addAnnotation(text: String, at position: SIMD3<Float>) {
        let anchor = AnchorEntity(world: position)
        
        // Create a simple text mesh
        let mesh = MeshResource.generateText(text,
                                           extrusionDepth: 0.01,
                                           font: .systemFont(ofSize: 0.05),
                                           containerFrame: .zero,
                                           alignment: .center,
                                           lineBreakMode: .byCharWrapping)
        
        let material = SimpleMaterial(color: .white, isMetallic: false)
        let entity = ModelEntity(mesh: mesh, materials: [material])
        
        // Billboard constraint so text always faces camera
        entity.look(at: arView.cameraTransform.translation, from: position, relativeTo: nil)
        
        anchor.addChild(entity)
        arView.scene.addAnchor(anchor)
    }
}
