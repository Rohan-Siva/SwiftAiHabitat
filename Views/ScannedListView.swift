import SwiftUI

struct ScannedListView: View {
    @EnvironmentObject var arViewModel: ARViewModel
    
    var body: some View {
        NavigationView {
            List {
                ForEach(arViewModel.detectedObjects) { object in
                    HStack {
                        VStack(alignment: .leading) {
                            Text(object.name)
                                .font(.headline)
                            Text(object.category)
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        Spacer()
                        Text(String(format: "(%.1f, %.1f)", object.position.x, object.position.z))
                            .font(.caption)
                            .padding(5)
                            .background(Color.gray.opacity(0.2))
                            .cornerRadius(5)
                    }
                }
                .onDelete { indexSet in
                    arViewModel.detectedObjects.remove(atOffsets: indexSet)
                }
            }
            .navigationTitle("Scanned Items")
            .overlay(
                Group {
                    if arViewModel.detectedObjects.isEmpty {
                        VStack {
                            Image(systemName: "cube.transparent")
                                .font(.largeTitle)
                                .foregroundColor(.gray)
                            Text("No objects scanned yet")
                                .foregroundColor(.gray)
                        }
                    }
                }
            )
        }
    }
}
