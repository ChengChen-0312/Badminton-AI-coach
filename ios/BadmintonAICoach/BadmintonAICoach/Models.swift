import Foundation

struct CreateJobRequest: Codable {
    let video_url: String
    let video_path: String?
    let config_profile: String
    let enable_llm: Bool
}

struct CreateJobResponse: Codable {
    let job_id: String
    let status: String
}

struct JobStatusResponse: Codable {
    let job_id: String
    let status: String
    let progress: Double?
    let error: String?
}

struct CourtDetectionReport: Codable {
    let source: String?
    let confidence: Double?
    let reason: String?
}

struct StrokeClassifierReport: Codable {
    let enabled: Bool?
    let mode: String?
    let checkpoint: String?
}

struct SummaryReport: Codable {
    let overall_score: Double?
    let confidence: Double?
}

struct StrokeReportItem: Codable, Identifiable {
    let id = UUID()
    let final_type: String?
    let confidence: Double?
    let landing_region: String?
    let hitter_role: String?

    private enum CodingKeys: String, CodingKey {
        case final_type
        case confidence
        case landing_region
        case hitter_role
    }
}

struct AnalysisReportResponse: Codable {
    let court_detection: CourtDetectionReport?
    let stroke_classifier: StrokeClassifierReport?
    let stroke_count: Int?
    let summary: SummaryReport?
    let strokes: [StrokeReportItem]?
    let label_space_version: String?
}
