from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
from typing import Iterable

from .config import IndexerConfig
from .db import IndexDB
from .exporter import export_jsonl
from .face import FACE_SCAN_VERSION, FaceScanError, FaceScanResult, scan_face_status
from .ffmpeg_tools import detect_scene_cuts, probe_media, scene_cuts_to_shots
from .fingerprint import fingerprint_file
from .models import TimeRange, VisionAnalysis
from .qc import write_qc_report
from .sampling import extract_representative_frames, is_nearly_black
from .segmentation import subdivide_shot
from .utils import relpath_for_display, stable_source_uid
from .vision import OllamaError, analyze_clip


class IndexingPipeline:
    def __init__(self, cfg: IndexerConfig, workspace: Path, logger: logging.Logger):
        self.cfg = cfg
        self.workspace = workspace.resolve()
        self.logger = logger
        self.frames_root = self.workspace / "frames"
        self.exports_dir = self.workspace / "exports"
        self.db = IndexDB(self.workspace / "footage_index.db")
        self.index_signature = cfg.index_signature()
        self.legacy_v121_signature = cfg.legacy_v121_index_signature()

    def close(self) -> None:
        self.db.close()

    def discover(self, input_path: Path) -> tuple[Path, list[Path]]:
        input_path = input_path.resolve()
        if input_path.is_file():
            if input_path.suffix.lower() not in self.cfg.supported_extensions:
                raise ValueError(f"Unsupported video extension: {input_path.suffix}")
            return input_path.parent, [input_path]
        if not input_path.is_dir():
            raise FileNotFoundError(f"Input path not found: {input_path}")
        files = [
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in self.cfg.supported_extensions
        ]
        files.sort(key=lambda p: p.as_posix().lower())
        return input_path, files

    def run(self, input_path: Path, *, force: bool = False, limit: int | None = None, no_ai: bool = False) -> dict:
        root, files = self.discover(input_path)
        if limit is not None:
            files = files[:max(0, limit)]
        if not files:
            raise RuntimeError("No supported video files found")
        self.logger.info("Found %d source video(s)", len(files))

        summary = {"sources_total": len(files), "sources_indexed": 0, "sources_partial": 0, "sources_skipped": 0}
        for source in files:
            status = self._process_source(source, root, force=force, no_ai=no_ai)
            if status == "SKIPPED":
                summary["sources_skipped"] += 1
            elif status == "INDEXED":
                summary["sources_indexed"] += 1
            else:
                summary["sources_partial"] += 1

        exported = export_jsonl(self.db, self.exports_dir / "clips.jsonl")
        report = write_qc_report(self.db, self.workspace / "qc_report.json")
        summary["exported_clips"] = exported
        summary["qc_passed"] = report["passed"]
        (self.workspace / "last_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _save_face_for_paths(self, clip_id: int, times: list[float], paths: list[Path]) -> str:
        try:
            result = scan_face_status(paths, self.cfg)
            evidence = result.evidence_payload(times)
            self.db.save_face_scan(
                clip_id, status=result.face_status, version=FACE_SCAN_VERSION,
                mode=result.mode, evidence=evidence,
            )
            return result.face_status
        except Exception as exc:
            # Face metadata must never destroy a completed semantic index. Because Agent 2
            # accepts `uncertain`, scanner failures degrade safely instead of failing the clip.
            self.logger.warning("    face scan uncertain: %s", exc)
            self.db.save_face_scan(
                clip_id, status="uncertain", version=FACE_SCAN_VERSION,
                mode="representative_frames_scan_error",
                evidence={"reason": str(exc)[:500], "frames": []},
            )
            return "uncertain"

    def _face_paths_for_row(self, row) -> tuple[list[float], list[Path]]:
        times = [float(x) for x in json.loads(row["representative_times_json"] or "[]")]
        paths = [Path(x) for x in json.loads(row["representative_paths_json"] or "[]")]
        if paths and all(p.exists() and p.stat().st_size > 0 for p in paths):
            return times, paths

        source = Path(row["source_absolute_path"])
        segment = TimeRange(float(row["start_time"]), float(row["end_time"]))
        self.logger.info("    representative frame cache missing for %s -> regenerate only frames", row["clip_uid"])
        return extract_representative_frames(
            source, segment, row["clip_uid"], row["source_uid"], self.frames_root, self.cfg
        )

    def _backfill_source_faces(self, source_id: int, *, force: bool = False) -> dict[str, int]:
        rows = self.db.indexed_clips_needing_face_scan(FACE_SCAN_VERSION, source_id=source_id, force=force)
        counts = {"scanned": 0, "no_face": 0, "face_visible": 0, "uncertain": 0}
        for row in rows:
            try:
                times, paths = self._face_paths_for_row(row)
                status = self._save_face_for_paths(int(row["id"]), times, paths)
            except Exception as exc:
                self.logger.warning("    face backfill frame failure for %s: %s", row["clip_uid"], exc)
                self.db.save_face_scan(
                    int(row["id"]), status="uncertain", version=FACE_SCAN_VERSION,
                    mode="representative_frames_frame_error",
                    evidence={"reason": str(exc)[:500], "frames": []},
                )
                status = "uncertain"
            counts["scanned"] += 1
            counts[status] += 1
        return counts

    def backfill_faces(self, *, force: bool = False) -> dict:
        rows = self.db.indexed_clips_needing_face_scan(FACE_SCAN_VERSION, force=force)
        total_indexed = len(self.db.all_indexed_clips())
        self.logger.info("Face backfill: %d indexed clip(s), %d need scan", total_indexed, len(rows))
        counts = {"scanned": 0, "no_face": 0, "face_visible": 0, "uncertain": 0}
        for i, row in enumerate(rows, start=1):
            if i == 1 or i % 10 == 0 or i == len(rows):
                self.logger.info("  face-backfill %d/%d", i, len(rows))
            try:
                times, paths = self._face_paths_for_row(row)
                status = self._save_face_for_paths(int(row["id"]), times, paths)
            except Exception as exc:
                self.logger.warning("  face backfill frame failure for %s: %s", row["clip_uid"], exc)
                self.db.save_face_scan(
                    int(row["id"]), status="uncertain", version=FACE_SCAN_VERSION,
                    mode="representative_frames_frame_error",
                    evidence={"reason": str(exc)[:500], "frames": []},
                )
                status = "uncertain"
            counts["scanned"] += 1
            counts[status] += 1

        exported = export_jsonl(self.db, self.exports_dir / "clips.jsonl")
        summary = {
            "agent1_face_scan_version": FACE_SCAN_VERSION,
            "indexed_clips_total": total_indexed,
            "clips_scanned": counts["scanned"],
            "face_status": {
                "no_face": counts["no_face"],
                "face_visible": counts["face_visible"],
                "uncertain": counts["uncertain"],
            },
            "exported_clips": exported,
        }
        (self.workspace / "last_face_backfill_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    def _analysis_paths_for_row(self, row) -> tuple[list[float], list[Path]]:
        """Reuse cached representative frames for editorial analysis; regenerate only when missing."""
        times = [float(x) for x in json.loads(row["representative_times_json"] or "[]")]
        paths = [Path(x) for x in json.loads(row["representative_paths_json"] or "[]")]
        if paths and all(p.exists() and p.stat().st_size > 0 for p in paths):
            return times, paths
        source = Path(row["source_absolute_path"])
        segment = TimeRange(float(row["start_time"]), float(row["end_time"]))
        self.logger.info("  editorial frame cache missing for %s -> regenerate representative frames only", row["clip_uid"])
        times, paths = extract_representative_frames(
            source, segment, row["clip_uid"], row["source_uid"], self.frames_root, self.cfg
        )
        self.db.save_frame_evidence(int(row["id"]), times, paths)
        # save_frame_evidence marks FRAMES_READY; preserve an existing usable index until the
        # replacement v1.3 analysis succeeds.
        self.db.conn.execute("UPDATE clips SET status='INDEXED' WHERE id=?", (int(row["id"]),))
        self.db.conn.commit()
        return times, paths

    def _upgrade_editorial_rows(self, rows) -> dict[str, int]:
        counts = {"requested": len(rows), "upgraded": 0, "failed": 0, "frames_regenerated": 0}
        for i, row in enumerate(rows, start=1):
            if i == 1 or i % 10 == 0 or i == len(rows):
                self.logger.info("  editorial-backfill %d/%d", i, len(rows))
            try:
                cached_paths = [Path(x) for x in json.loads(row["representative_paths_json"] or "[]")]
                had_cache = bool(cached_paths) and all(p.exists() and p.stat().st_size > 0 for p in cached_paths)
                times, paths = self._analysis_paths_for_row(row)
                if not had_cache:
                    counts["frames_regenerated"] += 1
                if self.cfg.skip_nearly_black_clips and is_nearly_black(
                    paths, self.cfg.black_mean_threshold, self.cfg.black_std_threshold
                ):
                    analysis = VisionAnalysis(
                        description="Mostly black or blank imagery with no clearly usable visual subject.",
                        primary_subject="none",
                        content_type="other",
                        quality_flags=["mostly_black_or_blank"],
                        usable=False,
                        exclusion_reason="mostly_black_or_blank",
                        confidence_level="high",
                        analysis_confidence=0.95,
                        confidence_basis=["deterministic black/blank frame check"],
                        camera_motion="unknown",
                        editorial_role="other",
                        visual_family="other__other__unknown__unknown__unknown__unknown",
                        visual_quality_score=0.10,
                        editorial_usefulness_score=0.0,
                    )
                    model = "deterministic_quality_check"
                else:
                    analysis = analyze_clip(paths, self.cfg)
                    model = self.cfg.vision_model
                self.db.save_analysis(
                    int(row["id"]), analysis, model, self.cfg.prompt_version, self.cfg.schema_version
                )
                counts["upgraded"] += 1
            except Exception as exc:
                # Preserve the old indexed analysis and face metadata. Editorial backfill is an
                # enhancement pass; a single model/format failure must not damage a working library.
                self.logger.warning("  editorial backfill failed for %s; keeping previous analysis: %s", row["clip_uid"], exc)
                self.db.conn.execute("UPDATE clips SET status='INDEXED' WHERE id=?", (int(row["id"]),))
                self.db.conn.commit()
                counts["failed"] += 1
        return counts

    def backfill_editorial(self, *, force: bool = False, limit: int | None = None) -> dict:
        rows = self.db.indexed_clips_needing_editorial(
            self.cfg.schema_version, self.cfg.prompt_version, force=force
        )
        if limit is not None:
            rows = rows[:max(0, limit)]
        total_indexed = len(self.db.all_indexed_clips())
        self.logger.info(
            "Editorial backfill v1.3: %d indexed clip(s), %d need editorial vision upgrade",
            total_indexed, len(rows),
        )
        counts = self._upgrade_editorial_rows(rows)

        upgraded_sources = 0
        for src in self.db.all_sources():
            source_id = int(src["id"])
            if self.db.source_editorial_complete(source_id, self.cfg.schema_version, self.cfg.prompt_version):
                self.db.update_source_index_signature(source_id, self.index_signature)
                upgraded_sources += 1

        exported = export_jsonl(self.db, self.exports_dir / "clips.jsonl")
        report = write_qc_report(self.db, self.workspace / "qc_report.json")
        current_rows = [
            r for r in self.db.all_indexed_clips()
            if int(r["analysis_version"] or 0) >= self.cfg.schema_version and r["prompt_version"] == self.cfg.prompt_version
        ]
        quality_scores: list[float] = []
        usefulness_scores: list[float] = []
        families: set[str] = set()
        roles: dict[str, int] = {}
        for row in current_rows:
            try:
                analysis = json.loads(row["analysis_json"] or "{}")
            except Exception:
                continue
            q = analysis.get("visual_quality_score")
            u = analysis.get("editorial_usefulness_score")
            if isinstance(q, (int, float)):
                quality_scores.append(float(q))
            if isinstance(u, (int, float)):
                usefulness_scores.append(float(u))
            fam = str(analysis.get("visual_family", "unknown"))
            if fam and fam != "unknown":
                families.add(fam)
            role = str(analysis.get("editorial_role", "other"))
            roles[role] = roles.get(role, 0) + 1

        summary = {
            "agent1_version": "1.3.0",
            "editorial_profile_version": 1,
            "indexed_clips_total": total_indexed,
            "clips_requested": counts["requested"],
            "clips_upgraded": counts["upgraded"],
            "clips_failed_preserved": counts["failed"],
            "representative_frames_regenerated": counts["frames_regenerated"],
            "sources_current": upgraded_sources,
            "visual_families": len(families),
            "editorial_roles": roles,
            "average_visual_quality": round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else None,
            "average_editorial_usefulness": round(sum(usefulness_scores) / len(usefulness_scores), 3) if usefulness_scores else None,
            "exported_clips": exported,
            "qc_passed": report["passed"],
        }
        (self.workspace / "last_editorial_backfill_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    def _process_source(self, source: Path, root: Path, *, force: bool, no_ai: bool) -> str:
        relative = relpath_for_display(source, root)
        source_uid = stable_source_uid(relative)
        stat = source.stat()
        self.logger.info("Source: %s", relative)

        existing = self.db.get_source(source_uid)
        # Fast resume/skip path: size + mtime + signature are unchanged.
        if (
            existing is not None and not force
            and int(existing["file_size"]) == stat.st_size
            and int(existing["mtime_ns"]) == stat.st_mtime_ns
            and existing["index_signature"] == self.index_signature
            and existing["status"] == "INDEXED"
        ):
            self.logger.info("  unchanged and already indexed -> skip")
            return "SKIPPED"

        media = probe_media(source)
        fingerprint = fingerprint_file(source, self.cfg.fingerprint_mode, self.cfg.fingerprint_sample_bytes)
        existing = self.db.get_source(source_uid)
        existing_signature = str(existing["index_signature"]) if existing is not None else ""
        legacy_editorial_upgrade = (
            existing is not None and not force
            and existing["fingerprint"] == fingerprint
            and existing_signature == self.legacy_v121_signature
        )
        same_content = (
            existing is not None and existing["fingerprint"] == fingerprint
            and existing_signature in {self.index_signature, self.legacy_v121_signature}
        )
        signature_to_store = existing_signature if legacy_editorial_upgrade else self.index_signature

        source_id = self.db.upsert_source(
            source_uid=source_uid,
            relative_path=relative,
            absolute_path=str(source),
            file_size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            fingerprint=fingerprint,
            fingerprint_mode=self.cfg.fingerprint_mode,
            index_signature=signature_to_store,
            media=media,
            status="PROBED",
        )

        # If content/config changed, regenerate the deterministic segmentation plan.
        if force or not same_content or not self.db.clips_for_source(source_id):
            self.logger.info("  detecting physical shots...")
            self.db.reset_source_children(source_id)
            frame_source_dir = self.frames_root / source_uid
            if frame_source_dir.exists():
                shutil.rmtree(frame_source_dir, ignore_errors=True)
            cuts = detect_scene_cuts(source, self.cfg.scene_threshold, media.duration)
            shots = scene_cuts_to_shots(cuts, media.duration, self.cfg.min_shot_sec)
            self.logger.info("  physical shots: %d", len(shots))
            self._store_segmentation(source, source_id, source_uid, shots)
        else:
            self.logger.info("  resuming existing segmentation")

        if legacy_editorial_upgrade and self.db.clips_for_source(source_id):
            self.logger.info("  v1.2.1 index detected -> preserve segmentation and upgrade editorial vision only")
            rows = self.db.indexed_clips_needing_editorial(
                self.cfg.schema_version, self.cfg.prompt_version, source_id=source_id, force=False
            )
            result = self._upgrade_editorial_rows(rows)
            if result["failed"] == 0 and self.db.source_editorial_complete(
                source_id, self.cfg.schema_version, self.cfg.prompt_version
            ):
                self.db.update_source_index_signature(source_id, self.index_signature)
                self.logger.info("  editorial vision upgrade complete; segmentation unchanged")
            elif result["failed"]:
                self.logger.warning("  editorial vision upgrade preserved %d old analysis record(s) for retry", result["failed"])

        self.db.set_source_status(source_id, "ANALYZING")
        pending = self.db.pending_clips_for_source(source_id)
        self.logger.info("  clips pending: %d", len(pending))
        for row in pending:
            clip_uid = row["clip_uid"]
            segment = TimeRange(float(row["start_time"]), float(row["end_time"]))
            self.logger.info("    %s %.2f-%.2f", clip_uid, segment.start, segment.end)
            try:
                times = [float(x) for x in json.loads(row["representative_times_json"] or "[]")]
                paths = [Path(x) for x in json.loads(row["representative_paths_json"] or "[]")]
                if not paths or not all(p.exists() and p.stat().st_size > 0 for p in paths):
                    times, paths = extract_representative_frames(
                        source, segment, clip_uid, source_uid, self.frames_root, self.cfg
                    )
                    self.db.save_frame_evidence(int(row["id"]), times, paths)
                else:
                    self.logger.info("      reusing %d cached representative frame(s)", len(paths))
            except Exception as exc:
                self.logger.exception("    frame extraction failed")
                self.db.fail_clip(int(row["id"]), "FRAME_EXTRACTION_FAILED", str(exc))
                continue

            if no_ai:
                self.db.fail_clip(int(row["id"]), "FRAMES_READY", "AI analysis intentionally skipped with --no-ai")
                continue

            if self.cfg.skip_nearly_black_clips and is_nearly_black(
                paths, self.cfg.black_mean_threshold, self.cfg.black_std_threshold
            ):
                analysis = VisionAnalysis(
                    description="Mostly black or blank imagery with no clearly usable visual subject.",
                    primary_subject="none",
                    content_type="other",
                    quality_flags=["mostly_black_or_blank"],
                    usable=False,
                    exclusion_reason="mostly_black_or_blank",
                    confidence_level="high",
                    analysis_confidence=0.95,
                    confidence_basis=["deterministic black/blank frame check"],
                    camera_motion="unknown",
                    editorial_role="other",
                    visual_family="other__other__unknown__unknown__unknown__unknown",
                    visual_quality_score=0.10,
                    editorial_usefulness_score=0.0,
                )
                self.db.save_analysis(
                    int(row["id"]), analysis, "deterministic_quality_check",
                    self.cfg.prompt_version, self.cfg.schema_version,
                )
                self.db.save_face_scan(
                    int(row["id"]), status="no_face", version=FACE_SCAN_VERSION,
                    mode="deterministic_black_blank",
                    evidence={"reason": "Clip is deterministically black/blank.", "frames": []},
                )
                continue

            try:
                analysis = analyze_clip(paths, self.cfg)
                self.db.save_analysis(
                    int(row["id"]), analysis, self.cfg.vision_model,
                    self.cfg.prompt_version, self.cfg.schema_version,
                )
                self._save_face_for_paths(int(row["id"]), times, paths)
            except OllamaError as exc:
                self.logger.error("    model failed: %s", exc)
                self.db.fail_clip(int(row["id"]), "MODEL_FAILED", str(exc))
            except Exception as exc:
                self.logger.exception("    validation/unexpected failure")
                self.db.fail_clip(int(row["id"]), "VALIDATION_FAILED", str(exc))

        # Resumed PARTIAL sources may contain already-indexed v1.1.x clips without face metadata.
        self._backfill_source_faces(source_id)

        counts = self.db.source_counts(source_id)
        if counts["clips"] > 0 and counts["failed"] == 0:
            self.db.set_source_status(source_id, "INDEXED", indexed=True)
            self.logger.info("  complete: %d shots / %d clips", counts["shots"], counts["clips"])
            return "INDEXED"
        self.db.set_source_status(source_id, "PARTIAL", error=f"{counts['failed']} clip(s) incomplete")
        self.logger.warning("  partial: %d/%d clips indexed", counts["indexed"], counts["clips"])
        return "PARTIAL"

    def _store_segmentation(self, source: Path, source_id: int, source_uid: str, shots: Iterable[TimeRange]) -> None:
        clip_index = 0
        for shot_index, shot in enumerate(shots, start=1):
            shot_uid = f"{source_uid}_shot_{shot_index:05d}"
            shot_id = self.db.add_shot(source_id, shot_uid, shot_index, shot, "ffmpeg_scene_score")
            try:
                semantic = subdivide_shot(source, shot, self.cfg)
            except Exception as exc:
                # Deterministic fallback: if adaptive probing fails, preserve coverage via even chunks.
                self.logger.warning("  adaptive subdivision failed for %s; using bounded chunks: %s", shot_uid, exc)
                semantic = []
                cursor = shot.start
                while cursor < shot.end - 0.05:
                    end = min(shot.end, cursor + self.cfg.max_semantic_clip_sec)
                    semantic.append(TimeRange(cursor, end))
                    cursor = end
            for seg in semantic:
                clip_index += 1
                clip_uid = f"{source_uid}_clip_{clip_index:06d}"
                self.db.add_clip(source_id, shot_id, clip_uid, clip_index, seg)
