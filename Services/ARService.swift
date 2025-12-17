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
    
    
    private var selectedEntity: ModelEntity?
    
    func addVirtualObject(at position: SIMD3<Float>, label: String = "Object") {
        let anchor = AnchorEntity(world: position)
        
        let color = UIColor(hue: CGFloat.random(in: 0...1), saturation: 0.8, brightness: 0.8, alpha: 1.0)
        let mesh = MeshResource.generateBox(size: 0.15)
        let material = SimpleMaterial(color: color, isMetallic: false)
        let entity = ModelEntity(mesh: mesh, materials: [material])
        
        entity.generateCollisionShapes(recursive: true)
        entity.name = label
        
        anchor.addChild(entity)
        arView.scene.addAnchor(anchor)
    }
    
    func selectEntity(at point: CGPoint) -> String? {
        let hitTest = arView.hitTest(point)
        if let firstHit = hitTest.first, let entity = firstHit.entity as? ModelEntity {
            selectedEntity = entity
            
            let material = SimpleMaterial(color: .green, isMetallic: true)
            entity.model?.materials = [material]
            
            return entity.name
        }
        return nil
    }
    
    func moveSelectedEntity(to point: CGPoint) {
        guard let entity = selectedEntity, let result = raycast(from: point) else { return }
        
        let position = SIMD3<Float>(result.worldTransform.columns.3.x,
                                  result.worldTransform.columns.3.y,
                                  result.worldTransform.columns.3.z)
        
        if let anchor = entity.parent as? AnchorEntity {
            anchor.position = position
        }
    }
    
    func deselectEntity() {
        guard let entity = selectedEntity else { return }
        let material = SimpleMaterial(color: .blue, isMetallic: false)
        entity.model?.materials = [material]
        selectedEntity = nil
    }

    
    func raycast(from point: CGPoint) -> ARRaycastResult? {
        let query = arView.makeRaycastQuery(from: point, allowing: .estimatedPlane, alignment: .any)
        guard let raycastQuery = query else { return nil }
        
        let results = arView.session.raycast(raycastQuery)
        return results.first
    }
    
    func toggleMeshVisualization(_ enabled: Bool) {
        if enabled {
            arView.debugOptions.insert([.showSceneUnderstanding, .showPhysics])
            arView.environment.sceneUnderstanding.options.insert(.occlusion)
        } else {
            arView.debugOptions.remove([.showSceneUnderstanding, .showPhysics])
            arView.environment.sceneUnderstanding.options.remove(.occlusion)
        }
    }
    
    func addAnnotation(text: String, at position: SIMD3<Float>) {
        let anchor = AnchorEntity(world: position)
        
        let mesh = MeshResource.generateText(text,
                                           extrusionDepth: 0.01,
                                           font: .systemFont(ofSize: 0.05),
                                           containerFrame: .zero,
                                           alignment: .center,
                                           lineBreakMode: .byCharWrapping)
        
        let material = SimpleMaterial(color: .white, isMetallic: false)
        let entity = ModelEntity(mesh: mesh, materials: [material])
        
        entity.look(at: arView.cameraTransform.translation, from: position, relativeTo: nil)
        
        anchor.addChild(entity)
        arView.scene.addAnchor(anchor)
    }
}
