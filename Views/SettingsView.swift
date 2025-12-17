import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var arViewModel: ARViewModel
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Inference")) {
                    Text("Model: SwiftAIHabitatViT")
                    Text("Version: 1.0")
                    Text("Provider: Core ML + Neural Engine")
                }
                
                Section(header: Text("Visualization")) {
                    Toggle("Debug Mesh", isOn: $arViewModel.isDebugMode)
                }
                
                Section(header: Text("Data")) {
                    Button("Clear All Scans") {
                        arViewModel.resetSession()
                    }
                    .foregroundColor(.red)
                }
            }
            .navigationTitle("Settings")
        }
    }
}
