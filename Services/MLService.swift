import CoreML
import Vision
import UIKit

class MLService: ObservableObject {
    private var model: VNCoreMLModel?
    
    @Published var lastPredictionLabel: String = ""
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        do {
            let config = MLModelConfiguration()
            let coreMLModel = try SwiftAIHabitatTransformer(configuration: config)
            self.model = try VNCoreMLModel(for: coreMLModel.model)
            print("MLService: Model loaded successfully.")
        } catch {
            print("MLService: Failed to load model. \(error)")
        }
    }
    
    func predictAffordances(for pixelBuffer: CVPixelBuffer) async -> String {
        guard let model = model else { return "Model not ready" }
        
        return await withCheckedContinuation { continuation in
            let request = VNCoreMLRequest(model: model) { [weak self] request, error in
                if let error = error {
                    print("Prediction error: \(error)")
                    continuation.resume(returning: "Error")
                    return
                }
                
                guard let results = request.results as? [VNClassificationObservation],
                      let topResult = results.first else {
                    continuation.resume(returning: "Unknown")
                    return
                }
                
                let prediction = "\(topResult.identifier) (\(Int(topResult.confidence * 100))%)"
                DispatchQueue.main.async {
                    self?.lastPredictionLabel = prediction
                }
                continuation.resume(returning: prediction)
            }
            
            request.imageCropAndScaleOption = .centerCrop
            
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            do {
                try handler.perform([request])
            } catch {
                print("Failed to perform request: \(error)")
                continuation.resume(returning: "Error")
            }
        }
    }
    
    func queryFoundationModel(prompt: String) async -> String {
       
        try? await Task.sleep(nanoseconds: 500_000_000) 
        
        if prompt.lowercased().contains("move") {
            return "Moving this object might block the path."
        } else if prompt.lowercased().contains("what") {
            return "This appears to be a \(lastPredictionLabel.isEmpty ? "scanned object" : lastPredictionLabel)."
        } else {
            return "I can help analyze spatial layout and object affordances."
        }
    }
}
