import Foundation

class PersistenceService {
    static let shared = PersistenceService()
    private let saveKey = "scanned_objects_v1"
    
    func save(_ objects: [ScannedObject]) {
        do {
            let data = try JSONEncoder().encode(objects)
            UserDefaults.standard.set(data, forKey: saveKey)
        } catch {
            print("Failed to save objects: \(error)")
        }
    }
    
    func load() -> [ScannedObject] {
        guard let data = UserDefaults.standard.data(forKey: saveKey) else { return [] }
        do {
            return try JSONDecoder().decode([ScannedObject].self, from: data)
        } catch {
            print("Failed to load objects: \(error)")
            return []
        }
    }
    
    func clear() {
        UserDefaults.standard.removeObject(forKey: saveKey)
    }
}
