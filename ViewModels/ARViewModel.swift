import SwiftUI
import ARKit
import Combine

class ARViewModel: ObservableObject {
    @Published var arService = ARService()
    @Published var visionService = VisionService()
    @Published var mlService = MLService()
    
    @Published var detectedObjects: [ScannedObject] = [] {
        didSet {
            PersistenceService.shared.save(detectedObjects)
        }
    }
    @Published var lastPrediction: String = ""
    @Published var isDebugMode: Bool = false {
        didSet {
            arService.toggleMeshVisualization(isDebugMode)
        }
    }
    
    private var cancellables = Set<AnyCancellable>()
    
    init() {
        self.detectedObjects = PersistenceService.shared.load()
        setupBindings()
    }
    
    private func setupBindings() {
        arService.$currentFrame
            .throttle(for: .seconds(1.0), scheduler: RunLoop.main, latest: true)
            .compactMap { $0 }
            .sink { [weak self] frame in
                self?.processFrame(frame)
            }
            .store(in: &cancellables)
    }
    
    private func processFrame(_ frame: ARFrame) {
        let pixelBuffer = frame.capturedImage
        
        Task {
            // 1. Run Text Recognition
            let recognizedText = await visionService.recognizeText(in: pixelBuffer)
            if !recognizedText.isEmpty {
                DispatchQueue.main.async {
                    self.lastPrediction = "Reading: \(recognizedText.prefix(3).joined(separator: ", "))"
                }
            }
            
            // 2. Run Vision Saliency (only if no text found to avoid noise)
            if recognizedText.isEmpty {
                let boundingBoxes = visionService.performSaliencyRequest(on: pixelBuffer)
                if !boundingBoxes.isEmpty {
                     DispatchQueue.main.async {
                        self.lastPrediction = "Object Detected"
                    }
                }
            }
        }
    }
    
    @Published var isWhatIfMode: Bool = false
    @Published var selectedObject: String?
    
    // ... setupBindings ... (unchanged)
    
    func handleTap(at point: CGPoint) {
        if isWhatIfMode {
            handleWhatIfTap(at: point)
        } else {
            handleAnalysisTap(at: point)
        }
    }
    
    private func handleWhatIfTap(at point: CGPoint) {
        // 1. Try to select an existing object
        if let objectName = arService.selectEntity(at: point) {
            self.selectedObject = objectName
            self.lastPrediction = "Selected: \(objectName)"
            return
        }
        
        // 2. If nothing selected, maybe we want to deselect?
        // For this demo, if we have a selected object and tap elsewhere, we move it there
        if selectedObject != nil {
            arService.moveSelectedEntity(to: point)
            arService.deselectEntity()
            self.selectedObject = nil
            self.lastPrediction = "Moved object"
        }
    }
    
    private func handleAnalysisTap(at point: CGPoint) {
        if let result = arService.raycast(from: point) {
             let position = SIMD3<Float>(result.worldTransform.columns.3.x,
                                       result.worldTransform.columns.3.y,
                                       result.worldTransform.columns.3.z)
            
            // Check if we hit an existing object first?
            // For now, always place new object or analyze
            
            // Query model first to see if we should place or just label
            // For demo: Always place a "Scanned Object" then analyze it
            
            arService.addVirtualObject(at: position, label: "Analyzing...")
            
            Task {
                // prediction using our new MLService
                // We need the screen center or crop at tap location properly?
                // For simplicity, we predict the whole frame or center crop.
                // Ideally we crop around the tap, but MLService currently takes the whole buffer.
                // Let's assume center crop focus for now.
                
                guard let frame = arService.currentFrame else { return }
                let prediction = await mlService.predictAffordances(for: frame.capturedImage)
                
                DispatchQueue.main.async {
                    // Update label of the last placed object
                    // In a real app we'd track the UUID of the entity we just added.
                    self.arService.addAnnotation(text: prediction, at: position)
                    self.lastPrediction = "Detected: \(prediction)"
                    
                    let scannedObj = ScannedObject(id: UUID(), name: prediction, category: "Furniture", position: position)
                    self.detectedObjects.append(scannedObj)
                }
            }
        }
    }
    
    func handleUserQuery(_ query: String) {
        Task {
            let response = await mlService.queryFoundationModel(prompt: query)
            DispatchQueue.main.async {
                self.lastPrediction = response
            }
        }
    }
    
    func resetSession() {
        arService.setupAR()
        detectedObjects.removeAll()
        PersistenceService.shared.clear()
        lastPrediction = ""
    }
}
