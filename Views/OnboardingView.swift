import SwiftUI
import AVFoundation

struct OnboardingView: View {
    @Binding var isOnboardingComplete: Bool
    
    var body: some View {
        VStack(spacing: 20) {
            Spacer()
            
            Image(systemName: "camera.viewfinder")
                .font(.system(size: 80))
                .foregroundColor(.blue)
            
            Text("Welcome to SwiftAI Habitat")
                .font(.largeTitle)
                .fontWeight(.bold)
                .multilineTextAlignment(.center)
            
            Text("Scan your room, detect objects, and interact with your environment using AI.")
                .font(.body)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
            
            Spacer()
            
            Button(action: {
                requestPermissions()
            }) {
                Text("Get Started")
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .cornerRadius(12)
            }
            .padding(.horizontal)
            .padding(.bottom, 40)
        }
    }
    
    func requestPermissions() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            DispatchQueue.main.async {
                if granted {
                    isOnboardingComplete = true
                } else {
                    // Handle permission denial
                    print("Camera permission denied")
                }
            }
        }
    }
}
