import SwiftUI

struct OverlayView: View {
    @ObservedObject var viewModel: ARViewModel
    @State private var queryText = ""
    
    var body: some View {
        VStack {
            // Top Bar
            HStack {
                Button(action: {
                    viewModel.resetSession()
                }) {
                    Label("Reset", systemImage: "arrow.counterclockwise")
                        .padding(8)
                        .background(.ultraThinMaterial)
                        .cornerRadius(8)
                }
                
                Spacer()
                
                Toggle("Debug", isOn: $viewModel.isDebugMode)
                    .labelsHidden()
                    .padding(8)
                    .background(.ultraThinMaterial)
                    .cornerRadius(8)
                    .overlay(
                        Text("Debug")
                            .font(.caption)
                            .offset(y: 20)
                    )
            }
            .padding()
            
            Spacer()
            
            // Prediction/Status Banner
            if !viewModel.lastPrediction.isEmpty {
                Text(viewModel.lastPrediction)
                    .padding()
                    .background(.ultraThinMaterial)
                    .cornerRadius(12)
                    .padding()
            }
            
            // Interaction Controls
            HStack {
                TextField("Ask about the room...", text: $queryText)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .padding(.horizontal)
                
                Button(action: {
                    viewModel.handleUserQuery(queryText)
                    queryText = ""
                }) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title)
                }
                .padding(.trailing)
            }
            .padding(.bottom, 20)
        }
        .environmentObject(viewModel)
    }
}
