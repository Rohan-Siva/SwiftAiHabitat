import Vision
import CoreImage
import UIKit

class VisionService: ObservableObject {
    
    func performSaliencyRequest(on pixelBuffer: CVPixelBuffer) -> [CGRect] {
        let request = VNGenerateObjectnessBasedSaliencyImageRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        
        do {
            try handler.perform([request])
            guard let result = request.results?.first as? VNSaliencyImageObservation else { return [] }
            
            // Return salient objects if available
            if let salientObjects = result.salientObjects {
                return salientObjects.map { $0.boundingBox }
            }
            return []
        } catch {
            print("Vision request failed: \(error)")
            return []
        }
    }
    
    func recognizeText(in pixelBuffer: CVPixelBuffer) async -> [String] {
        return await withCheckedContinuation { continuation in
            let request = VNRecognizeTextRequest { request, error in
                guard let observations = request.results as? [VNRecognizedTextObservation], error == nil else {
                    continuation.resume(returning: [])
                    return
                }
                
                let recognizedStrings = observations.compactMap { observation in
                    observation.topCandidates(1).first?.string
                }
                continuation.resume(returning: recognizedStrings)
            }
            
            request.recognitionLevel = .accurate
            
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            do {
                try handler.perform([request])
            } catch {
                print("Text recognition failed: \(error)")
                continuation.resume(returning: [])
            }
        }
    }
}
