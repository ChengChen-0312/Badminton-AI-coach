from __future__ import annotations

import argparse
import json
import threading
import traceback
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.pipeline.analyse_video import analyse_video
from src.pipeline.extract_strokes import (
    infer_segment_frame_ranges_from_analysis,
    summarise_strokes_from_analysis,
    stroke_summaries_to_dicts,
)
from src.pipeline.label_space import build_label_space_metadata
from src.pipeline.report_generator import generate_match_report
from src.vision.court_env_profile import resolve_local_path_from_url_or_path


PROFILE_TO_CONFIG = {
    "v3_realtime_ios": "src/config/v3_realtime.yaml",
    "v3_realtime": "src/config/v3_realtime.yaml",
    "v3_ai_score": "src/config/v3_ai_score.yaml",
}


@dataclass
class JobState:
    job_id: str
    status: str = "queued"
    progress: float = 0.0
    error: Optional[str] = None
    request: Optional[Dict[str, Any]] = None
    report_path: Optional[str] = None


def _json_response(handler: BaseHTTPRequestHandler, code: int, body: Dict[str, Any]) -> None:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    handler.send_response(int(code))
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _load_request_json(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    n = int(handler.headers.get("Content-Length", "0") or 0)
    raw = handler.rfile.read(n) if n > 0 else b"{}"
    if not raw:
        return {}
    obj = json.loads(raw.decode("utf-8"))
    return obj if isinstance(obj, dict) else {}


def _resolve_video_path(payload: Dict[str, Any], repo_root: Path) -> Optional[Path]:
    # Preferred local testing path (desktop/iOS simulator bridge).
    vp = str(payload.get("video_path", "") or "").strip()
    if vp:
        p = resolve_local_path_from_url_or_path(vp, repo_root=repo_root)
        if p is not None:
            return p

    # Contract field from iOS doc (video_url). Also accept plain local paths.
    vu = str(payload.get("video_url", "") or "").strip()
    if not vu:
        return None
    return resolve_local_path_from_url_or_path(vu, repo_root=repo_root)


def _choose_config_path(payload: Dict[str, Any], repo_root: Path) -> Path:
    profile = str(payload.get("config_profile", "v3_realtime_ios") or "v3_realtime_ios").strip()
    cfg_rel = PROFILE_TO_CONFIG.get(profile, PROFILE_TO_CONFIG["v3_realtime_ios"])
    p = Path(cfg_rel)
    if not p.is_absolute():
        p = repo_root / p
    return p


def _simple_summary(strokes: list[dict[str, Any]]) -> Dict[str, Any]:
    if not strokes:
        return {"overall_score": 0.0, "confidence": 0.0}
    confs = []
    for s in strokes:
        try:
            c = float(s.get("confidence"))
            confs.append(max(0.0, min(1.0, c)))
        except Exception:
            continue
    if not confs:
        return {"overall_score": 0.0, "confidence": 0.0}
    mean_conf = float(sum(confs) / max(1, len(confs)))
    return {
        "overall_score": round(100.0 * mean_conf, 1),
        "confidence": round(mean_conf, 3),
    }


class AnalysisAPIServer:
    def __init__(self, repo_root: Path, out_root: Path):
        self.repo_root = repo_root
        self.out_root = out_root
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, JobState] = {}
        self.lock = threading.Lock()

    def create_job(self, payload: Dict[str, Any]) -> JobState:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        state = JobState(job_id=job_id, status="queued", progress=0.0, request=dict(payload))
        with self.lock:
            self.jobs[job_id] = state
        t = threading.Thread(target=self._run_job, args=(job_id, dict(payload)), daemon=True)
        t.start()
        return state

    def get_job(self, job_id: str) -> Optional[JobState]:
        with self.lock:
            return self.jobs.get(job_id)

    def _update_job(self, job_id: str, **kwargs: Any) -> None:
        with self.lock:
            st = self.jobs.get(job_id)
            if st is None:
                return
            for k, v in kwargs.items():
                setattr(st, k, v)

    def _run_job(self, job_id: str, payload: Dict[str, Any]) -> None:
        try:
            self._update_job(job_id, status="running", progress=0.05, error=None)

            video_path = _resolve_video_path(payload, self.repo_root)
            if video_path is None:
                raise RuntimeError(
                    "video_path missing. For this local server, use video_path or file:// video_url."
                )
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")

            cfg_path = _choose_config_path(payload, self.repo_root)
            if not cfg_path.exists():
                raise FileNotFoundError(f"Config not found: {cfg_path}")

            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            if isinstance(cfg, dict):
                cfg["__config_dir__"] = str(cfg_path.parent)
            self._update_job(job_id, progress=0.20)

            analysis = analyse_video(str(video_path), config=cfg)
            self._update_job(job_id, progress=0.55)

            classifier_cfg = cfg.get("classifier", {}) if isinstance(cfg.get("classifier", {}), dict) else {}
            classifier_min_confidence = float(classifier_cfg.get("min_confidence", 0.0) or 0.0)
            expected_num_classes = classifier_cfg.get("expected_num_classes")
            try:
                expected_num_classes_i = int(expected_num_classes) if expected_num_classes is not None else None
            except Exception:
                expected_num_classes_i = None
            expected_class_names = classifier_cfg.get("class_names")

            classifier_outputs = None
            classifier_labels = None
            classifier_enabled = False
            classifier_ckpt = classifier_cfg.get("checkpoint")
            label_space_meta = build_label_space_metadata(
                None,
                version_hint=classifier_cfg.get("label_space_version"),
                expected_num_classes=expected_num_classes_i,
                expected_class_names=expected_class_names,
            )
            if classifier_ckpt:
                ck = Path(str(classifier_ckpt))
                if not ck.is_absolute():
                    ck = self.repo_root / ck
                if ck.exists():
                    try:
                        from src.pipeline.stroke_classifier_runtime import StrokeClassifierRuntime

                        runtime = StrokeClassifierRuntime.from_checkpoint(
                            checkpoint_path=ck,
                            device=str(classifier_cfg.get("device", "auto")),
                            frame_size=classifier_cfg.get("frame_size"),
                            num_frames=classifier_cfg.get("num_frames"),
                            topk=int(max(1, classifier_cfg.get("topk", 3))),
                        )
                        segs = infer_segment_frame_ranges_from_analysis(analysis)
                        classifier_outputs = runtime.predict_labels_for_segments_from_video(video_path, segs)
                        classifier_labels = [
                            str(item.get("label")) if isinstance(item, dict) and item.get("label") is not None else None
                            for item in classifier_outputs
                        ]
                        classifier_enabled = True
                        label_space_meta = build_label_space_metadata(
                            runtime.classes,
                            version_hint=classifier_cfg.get("label_space_version"),
                            expected_num_classes=expected_num_classes_i,
                            expected_class_names=expected_class_names,
                        )
                    except Exception:
                        classifier_outputs = None
                        classifier_labels = None

            spatial_cfg = cfg.get("spatial_logic", {}) if isinstance(cfg.get("spatial_logic", {}), dict) else {}
            pose_cfg = cfg.get("pose", {}) if isinstance(cfg.get("pose", {}), dict) else {}
            summaries = summarise_strokes_from_analysis(
                analysis,
                classifier_labels=classifier_labels,
                classifier_outputs=classifier_outputs,
                classifier_min_confidence=float(max(0.0, classifier_min_confidence)),
                enable_hitter_inference=bool(spatial_cfg.get("enable_hitter_inference", True)),
                hitter_distance_max=float(spatial_cfg.get("hitter_distance_max", 200.0)),
                pose_window=int(pose_cfg.get("window", 3)) if isinstance(pose_cfg.get("window", 3), int) else 3,
            )
            stroke_dicts = stroke_summaries_to_dicts(summaries)
            self._update_job(job_id, progress=0.75)

            out_dir = self.out_root / job_id
            out_dir.mkdir(parents=True, exist_ok=True)
            report_paths = generate_match_report(
                job_id,
                stroke_dicts,
                str(out_dir),
                enable_heatmap=True,
                enable_timeline=True,
            )
            json_path = Path(report_paths["json"])
            report_data = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(report_data, list):
                report_data = {"strokes": report_data}

            det = getattr(analysis, "court_detection", None)
            report_data["court_detection"] = det if isinstance(det, dict) else {}
            report_data["stroke_classifier"] = {
                "enabled": bool(classifier_enabled),
                "checkpoint": str(classifier_ckpt) if classifier_ckpt else None,
                "mode": "auto",
                "min_confidence": float(max(0.0, classifier_min_confidence)),
                "label_space": label_space_meta,
            }
            report_data["label_space_version"] = label_space_meta.get("version")
            court_env = getattr(analysis, "court_env", None)
            if isinstance(court_env, dict):
                report_data["court_env"] = court_env
            report_data["summary"] = _simple_summary(stroke_dicts)
            report_data["job_id"] = job_id
            json_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")

            self._update_job(
                job_id,
                status="finished",
                progress=1.0,
                report_path=str(json_path),
                error=None,
            )
        except Exception as exc:
            tb = traceback.format_exc(limit=8)
            self._update_job(
                job_id,
                status="failed",
                progress=1.0,
                error=f"{exc}\n{tb}",
            )


def build_handler(server_state: AnalysisAPIServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover
            return

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/v1/analysis/jobs":
                try:
                    payload = _load_request_json(self)
                    st = server_state.create_job(payload)
                    _json_response(self, 200, {"job_id": st.job_id, "status": st.status})
                except Exception as exc:
                    _json_response(self, 400, {"error": str(exc)})
                return
            _json_response(self, 404, {"error": "not_found"})

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.rstrip("/")
            if path.startswith("/v1/analysis/jobs/"):
                tail = path[len("/v1/analysis/jobs/") :]
                if "/" in tail:
                    job_id, sub = tail.split("/", 1)
                    if sub == "report":
                        st = server_state.get_job(job_id)
                        if st is None:
                            _json_response(self, 404, {"error": "job_not_found"})
                            return
                        if st.status != "finished" or not st.report_path:
                            _json_response(
                                self,
                                409,
                                {
                                    "error": "report_not_ready",
                                    "status": st.status,
                                    "progress": st.progress,
                                    "job_id": st.job_id,
                                },
                            )
                            return
                        p = Path(st.report_path)
                        if not p.exists():
                            _json_response(self, 500, {"error": "report_missing"})
                            return
                        try:
                            data = json.loads(p.read_text(encoding="utf-8"))
                            if isinstance(data, list):
                                data = {"strokes": data}
                            _json_response(self, 200, data if isinstance(data, dict) else {"raw": data})
                        except Exception as exc:
                            _json_response(self, 500, {"error": f"report_decode_error:{exc}"})
                        return
                else:
                    st = server_state.get_job(tail)
                    if st is None:
                        _json_response(self, 404, {"error": "job_not_found"})
                        return
                    _json_response(
                        self,
                        200,
                        {
                            "job_id": st.job_id,
                            "status": st.status,
                            "progress": float(st.progress),
                            "error": st.error,
                        },
                    )
                    return
            _json_response(self, 404, {"error": "not_found"})

    return Handler


def run_server(host: str, port: int, repo_root: Path, out_root: Path) -> None:
    server_state = AnalysisAPIServer(repo_root=repo_root, out_root=out_root)
    handler = build_handler(server_state)
    httpd = ThreadingHTTPServer((host, int(port)), handler)
    print(f"[API] listening on http://{host}:{port}")
    print("[API] endpoints: POST /v1/analysis/jobs, GET /v1/analysis/jobs/{job_id}, GET /v1/analysis/jobs/{job_id}/report")
    httpd.serve_forever()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Local analysis API server for iOS MVP integration.")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--repo-root", type=str, default=".")
    ap.add_argument("--out-dir", type=str, default="reports/api_jobs")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    run_server(args.host, int(args.port), repo_root=repo_root, out_root=out_dir.resolve())


if __name__ == "__main__":
    main()
