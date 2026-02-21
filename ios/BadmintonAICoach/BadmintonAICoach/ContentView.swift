import SwiftUI

@MainActor
final class AnalysisViewModel: ObservableObject {
    @Published var baseURL = "http://127.0.0.1:8765"
    @Published var videoURL = ""
    @Published var videoPath = ""
    @Published var configProfile = "v3_realtime_ios"
    @Published var enableLLM = false

    @Published var latestJobID = ""
    @Published var statusText = "Idle"
    @Published var progress: Double = 0.0
    @Published var report: AnalysisReportResponse?
    @Published var lastError: String?

    func createJob() async {
        let videoURLTrim = videoURL.trimmingCharacters(in: .whitespacesAndNewlines)
        let videoPathTrim = videoPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !videoURLTrim.isEmpty || !videoPathTrim.isEmpty else {
            lastError = "Please input video_url or video_path first."
            return
        }
        do {
            let api = try APIClient(baseURLString: baseURL)
            let out = try await api.createJob(
                videoURL: videoURLTrim,
                videoPath: videoPathTrim.isEmpty ? nil : videoPathTrim,
                configProfile: configProfile,
                enableLLM: enableLLM
            )
            latestJobID = out.job_id
            statusText = out.status
            progress = 0.0
            report = nil
            lastError = nil
        } catch {
            lastError = error.localizedDescription
        }
    }

    func refreshStatus() async {
        guard !latestJobID.isEmpty else {
            lastError = "No job id yet."
            return
        }
        do {
            let api = try APIClient(baseURLString: baseURL)
            let out = try await api.getJobStatus(jobID: latestJobID)
            statusText = out.status
            progress = max(0.0, min(1.0, out.progress ?? progress))
            if let err = out.error, !err.isEmpty {
                lastError = err
            }
        } catch {
            lastError = error.localizedDescription
        }
    }

    func fetchReport() async {
        guard !latestJobID.isEmpty else {
            lastError = "No job id yet."
            return
        }
        do {
            let api = try APIClient(baseURLString: baseURL)
            let out = try await api.getReport(jobID: latestJobID)
            report = out
            statusText = "finished"
            progress = 1.0
            lastError = nil
        } catch {
            lastError = error.localizedDescription
        }
    }
}

struct ContentView: View {
    @ObservedObject var viewModel: AnalysisViewModel

    var body: some View {
        NavigationStack {
            Form {
                Section("Server") {
                    TextField("Base URL", text: $viewModel.baseURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    TextField("Video URL (remote/file://)", text: $viewModel.videoURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    TextField("Video Path (local path)", text: $viewModel.videoPath)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    TextField("Config Profile", text: $viewModel.configProfile)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    Toggle("Enable LLM", isOn: $viewModel.enableLLM)
                }

                Section("Run") {
                    Button("Create Analysis Job") {
                        Task { await viewModel.createJob() }
                    }
                    .buttonStyle(.borderedProminent)

                    HStack {
                        Button("Refresh Status") {
                            Task { await viewModel.refreshStatus() }
                        }
                        Button("Fetch Report") {
                            Task { await viewModel.fetchReport() }
                        }
                    }

                    LabeledContent("Job ID", value: viewModel.latestJobID.isEmpty ? "-" : viewModel.latestJobID)
                    LabeledContent("Status", value: viewModel.statusText)
                    ProgressView(value: viewModel.progress)
                }

                if let rep = viewModel.report {
                    Section("Report") {
                        LabeledContent("Label Space", value: rep.label_space_version ?? "-")
                        LabeledContent("Stroke Count", value: String(rep.stroke_count ?? 0))
                        if let score = rep.summary?.overall_score {
                            LabeledContent("Overall Score", value: String(format: "%.1f", score))
                        }
                        if let conf = rep.summary?.confidence {
                            LabeledContent("Score Confidence", value: String(format: "%.2f", conf))
                        }
                        if let reason = rep.court_detection?.reason {
                            LabeledContent("Court Reason", value: reason)
                        }
                    }

                    if let strokes = rep.strokes, !strokes.isEmpty {
                        Section("Latest Strokes") {
                            ForEach(Array(strokes.prefix(8).enumerated()), id: \.offset) { idx, s in
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("#\(idx + 1) \(s.final_type ?? \"unknown\")")
                                        .font(.headline)
                                    Text("hitter=\(s.hitter_role ?? \"-\") landing=\(s.landing_region ?? \"-\")")
                                        .font(.caption)
                                    if let c = s.confidence {
                                        Text(String(format: "confidence=%.2f", c))
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                    }
                                }
                            }
                        }
                    }
                }

                if let err = viewModel.lastError {
                    Section("Error") {
                        Text(err)
                            .foregroundStyle(.red)
                            .font(.caption)
                    }
                }
            }
            .navigationTitle("Badminton AI Coach")
        }
    }
}

#Preview {
    ContentView(viewModel: AnalysisViewModel())
}
