import Foundation

enum APIClientError: Error, LocalizedError {
    case invalidURL
    case badStatusCode(Int, String)

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid API base URL."
        case .badStatusCode(let code, let body):
            return "HTTP \(code): \(body)"
        }
    }
}

final class APIClient {
    private let baseURL: URL

    init(baseURLString: String) throws {
        guard let url = URL(string: baseURLString.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            throw APIClientError.invalidURL
        }
        self.baseURL = url
    }

    func createJob(
        videoURL: String,
        videoPath: String?,
        configProfile: String,
        enableLLM: Bool
    ) async throws -> CreateJobResponse {
        let req = CreateJobRequest(
            video_url: videoURL,
            video_path: videoPath,
            config_profile: configProfile,
            enable_llm: enableLLM
        )
        return try await send(path: "/v1/analysis/jobs", method: "POST", body: req)
    }

    func getJobStatus(jobID: String) async throws -> JobStatusResponse {
        try await send(path: "/v1/analysis/jobs/\(jobID)", method: "GET", body: Optional<Int>.none)
    }

    func getReport(jobID: String) async throws -> AnalysisReportResponse {
        try await send(path: "/v1/analysis/jobs/\(jobID)/report", method: "GET", body: Optional<Int>.none)
    }

    private func send<T: Decodable, B: Encodable>(path: String, method: String, body: B?) async throws -> T {
        let url = baseURL.appendingPathComponent(path.trimmingCharacters(in: CharacterSet(charactersIn: "/")))
        var req = URLRequest(url: url)
        req.httpMethod = method
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let body {
            req.httpBody = try JSONEncoder().encode(body)
        }

        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        guard (200...299).contains(http.statusCode) else {
            let bodyText = String(data: data, encoding: .utf8) ?? "<non-utf8 body>"
            throw APIClientError.badStatusCode(http.statusCode, bodyText)
        }
        return try JSONDecoder().decode(T.self, from: data)
    }
}
