from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

from .config import IndexerConfig
from .ffmpeg_tools import FFmpegError, require_binaries
from .logging_utils import setup_logging
from .pipeline import IndexingPipeline
from .vision import check_ollama


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="Agent1_Indexing",
        description="FacelessRC Agent 1 - local shot/clip-centric semantic footage indexer",
    )
    p.add_argument("input", nargs="?", help="Video file or folder containing source footage")
    p.add_argument("--workspace", default="./index", help="Persistent index workspace (default: ./index)")
    p.add_argument("--config", help="Optional JSON config file")
    p.add_argument("--check", action="store_true", help="Run environment/model preflight checks and exit")
    p.add_argument("--force", action="store_true", help="Re-index even if footage is unchanged")
    p.add_argument("--limit", type=int, help="Process only the first N discovered videos")
    p.add_argument("--no-ai", action="store_true", help="Build segmentation/frame evidence but do not call Qwen")
    p.add_argument("--face-backfill", action="store_true", help="Add/update face metadata in an existing workspace without re-indexing footage")
    p.add_argument("--editorial-backfill", action="store_true", help="Upgrade existing indexed clips to the v1.3 editorial vision schema without re-segmenting footage")
    p.add_argument("--verbose", action="store_true")
    return p


def run_check(cfg: IndexerConfig) -> int:
    print("FacelessRC Agent 1 preflight")
    print(f"Python: {platform.python_version()}")
    if sys.version_info < (3, 11):
        print("FAIL: Python 3.11+ required")
        return 2
    print("OK: Python version")
    try:
        ffmpeg, ffprobe = require_binaries()
        print(f"OK: ffmpeg = {ffmpeg}")
        print(f"OK: ffprobe = {ffprobe}")
    except FFmpegError as exc:
        print(f"FAIL: {exc}")
        return 2
    ok, message = check_ollama(cfg)
    print(("OK: " if ok else "FAIL: ") + message)
    if not ok:
        return 2
    print(f"Index signature: {cfg.index_signature()[:16]}")
    print("PRECHECK PASSED")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = IndexerConfig.load(args.config)
    except Exception as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    if args.check:
        return run_check(cfg)
    if not args.input and not args.face_backfill and not args.editorial_backfill:
        print("Error: input video/folder is required unless --check, --face-backfill, or --editorial-backfill is used", file=sys.stderr)
        return 2

    workspace = Path(args.workspace).expanduser()
    logger, log_path = setup_logging(workspace / "logs", verbose=args.verbose)
    logger.info("Agent 1 starting")
    logger.info("Workspace: %s", workspace.resolve())
    logger.info("Config signature: %s", cfg.index_signature())

    pipeline = IndexingPipeline(cfg, workspace, logger)
    try:
        if args.face_backfill:
            summary = pipeline.backfill_faces(force=args.force)
            print("\n" + json.dumps(summary, indent=2))
            print(f"\nIndex DB: {workspace.resolve() / 'footage_index.db'}")
            print(f"JSONL:    {workspace.resolve() / 'exports' / 'clips.jsonl'}")
            print(f"Face summary: {workspace.resolve() / 'last_face_backfill_summary.json'}")
            print(f"Log:      {log_path.resolve()}")
            return 0
        if args.editorial_backfill:
            summary = pipeline.backfill_editorial(force=args.force, limit=args.limit)
            print("\n" + json.dumps(summary, indent=2))
            print(f"\nIndex DB: {workspace.resolve() / 'footage_index.db'}")
            print(f"JSONL:    {workspace.resolve() / 'exports' / 'clips.jsonl'}")
            print(f"QC:       {workspace.resolve() / 'qc_report.json'}")
            print(f"Editorial summary: {workspace.resolve() / 'last_editorial_backfill_summary.json'}")
            print(f"Log:      {log_path.resolve()}")
            return 0 if summary.get("clips_failed_preserved", 0) == 0 else 1
        summary = pipeline.run(Path(args.input).expanduser(), force=args.force, limit=args.limit, no_ai=args.no_ai)
        print("\n" + json.dumps(summary, indent=2))
        print(f"\nIndex DB: {workspace.resolve() / 'footage_index.db'}")
        print(f"JSONL:    {workspace.resolve() / 'exports' / 'clips.jsonl'}")
        print(f"QC:       {workspace.resolve() / 'qc_report.json'}")
        print(f"Log:      {log_path.resolve()}")
        return 0 if args.no_ai or summary.get("sources_partial", 0) == 0 else 1
    except KeyboardInterrupt:
        logger.warning("Interrupted. Progress already committed; rerun the same command to resume.")
        return 130
    except Exception as exc:
        logger.exception("Agent 1 failed")
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Log: {log_path.resolve()}", file=sys.stderr)
        return 1
    finally:
        pipeline.close()
