import SwiftUI

struct MainView: View {
    @StateObject var arViewModel = ARViewModel()
    
    var body: some View {
        TabView {
            ZStack {
                ARViewContainer()
                    .edgesIgnoringSafeArea(.all)
                    .onTapGesture { location in
                        arViewModel.handleTap(at: location)
                    }
                
                VStack {
                    // Mode Toggle
                    Picker("Mode", selection: $arViewModel.isWhatIfMode) {
                        Text("Analyze").tag(false)
                        Text("What-If").tag(true)
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    .padding()
                    .background(Material.thinMaterial)
                    .cornerRadius(8)
                    .padding(.top, 50)
                    .padding(.horizontal)
                    
                    Spacer()
                    
                    if !arViewModel.isWhatIfMode {
                        OverlayView(viewModel: arViewModel)
                    } else {
                        // What-If specific instructions
                        Text(arViewModel.selectedObject != nil ? "Tap to Move" : "Tap object to Select")
                            .font(.headline)
                            .padding()
                            .background(Material.thinMaterial)
                            .cornerRadius(10)
                            .padding(.bottom)
                    }
                }
            }
            .environmentObject(arViewModel)
            .tabItem {
                Label("Habitat", systemImage: "arkit")
            }
            
            ScannedListView()
                .environmentObject(arViewModel)
                .tabItem {
                    Label("Scans", systemImage: "list.bullet")
                }
            
            SettingsView()
                .environmentObject(arViewModel)
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
        }
    }
}
