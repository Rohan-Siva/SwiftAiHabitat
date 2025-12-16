import SwiftUI
import ARKit
import Combine

class ARViewModel: ObservableObject {
    @Published var arService = ARService()
    @Published var visionService = VisionService()
    @Published var mlService = MLService()
    
    @Published var detectedObjects: [ScannedObject] = []
    @Published var lastPrediction: String = ""
    @Published var isDebugMode: Bool = false {
        didSet {
            arService.toggleMeshVisualization(isDebugMode)
        }
    }
    
    private var cancellables = Set<AnyCancellable>()
    
    init() {
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
    
    func handleTap(at point: CGPoint) {
        if let result = arService.raycast(from: point) {
            // Place an anchor at the tapped location
            let position = SIMD3<Float>(result.worldTransform.columns.3.x,
                                      result.worldTransform.columns.3.y,
                                      result.worldTransform.columns.3.z)
            
            arService.addVirtualObject(at: position)
            
            // Simulate analyzing the tapped object
            Task {
                let prediction = await mlService.queryFoundationModel(prompt: "What is at this location?")
                DispatchQueue.main.async {
                    self.arService.addAnnotation(text: "Analyzed", at: position)
                    self.lastPrediction = prediction
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
        lastPrediction = ""
    }
}
