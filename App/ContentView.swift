import SwiftUI

struct ContentView: View {
    @AppStorage("isOnboardingComplete") var isOnboardingComplete: Bool = false
    
    var body: some View {
        if isOnboardingComplete {
            MainView()
        } else {
            OnboardingView(isOnboardingComplete: $isOnboardingComplete)
        }
    }
}
