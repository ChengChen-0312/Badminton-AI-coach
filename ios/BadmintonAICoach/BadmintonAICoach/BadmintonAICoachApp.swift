import SwiftUI

@main
struct BadmintonAICoachApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: AnalysisViewModel())
        }
    }
}
