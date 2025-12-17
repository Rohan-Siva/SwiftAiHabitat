import Foundation
import CoreGraphics

struct ScannedObject: Identifiable, Equatable, Codable {
    var id: UUID = UUID()
    var name: String
    var category: String
    var position: SIMD3<Float>
    
    var confidence: Float = 1.0
    var boundingBox: CGRect = .zero
    var affordances: [String] = []
    var timestamp: Date = Date()
}
