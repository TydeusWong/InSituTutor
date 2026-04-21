import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_video_path(ingest_manifest: Dict[str, Any]) -> Path:
    demos = ingest_manifest.get("demos", [])
    if not demos:
        raise ValueError("ingest manifest has no demos")
    video_path = demos[0].get("ingest_video_path")
    if not video_path:
        raise ValueError("ingest_video_path missing")
    p = Path(str(video_path))
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def run_ffmpeg_extract_wav(video_path: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extract failed: {proc.stderr.strip()[:500]}")


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def normalize_segments_from_funasr(result: Any) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    if isinstance(result, list) and result:
        result = result[0]
    if not isinstance(result, dict):
        return segments

    sentence_info = result.get("sentence_info")
    if isinstance(sentence_info, list) and sentence_info:
        for idx, item in enumerate(sentence_info, start=1):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            ts = item.get("timestamp")
            start_ms = 0.0
            end_ms = 0.0
            if isinstance(ts, list) and len(ts) >= 2:
                start_ms = _safe_float(ts[0], 0.0)
                end_ms = _safe_float(ts[1], start_ms)
            else:
                start_ms = _safe_float(item.get("start"), 0.0)
                end_ms = _safe_float(item.get("end"), start_ms)
            if end_ms < start_ms:
                end_ms = start_ms
            speaker = item.get("spk")
            if speaker is None:
                speaker = item.get("speaker")
            speaker = str(speaker).strip() if speaker is not None and str(speaker).strip() else "unknown"
            segments.append(
                {
                    "segment_id": f"seg_{idx:04d}",
                    "start_sec": round(start_ms / 1000.0, 3),
                    "end_sec": round(end_ms / 1000.0, 3),
                    "speaker": speaker,
                    "text": text,
                }
            )
        return segments

    # Fallback: single utterance, maybe no sentence-level timestamps.
    text = str(result.get("text", "")).strip()
    if text:
        segments.append(
            {
                "segment_id": "seg_0001",
                "start_sec": 0.0,
                "end_sec": 0.0,
                "speaker": "unknown",
                "text": text,
            }
        )
    return segments


def transcribe_with_funasr(
    wav_path: Path,
    model_name: str,
    vad_model: str,
    punc_model: str,
    spk_model: str,
    device: str,
) -> Dict[str, Any]:
    try:
        from funasr import AutoModel
    except Exception as exc:
        raise RuntimeError(
            "FunASR import failed. Please install funasr first, e.g. `pip install funasr` in your runtime env."
        ) from exc

    kwargs: Dict[str, Any] = {
        "model": model_name,
        "vad_model": vad_model,
        "punc_model": punc_model,
        "device": device,
    }
    if spk_model:
        kwargs["spk_model"] = spk_model
    model = AutoModel(**kwargs)

    # Different funasr versions expose slightly different keys; keep request simple and normalize output.
    result = model.generate(input=str(wav_path), batch_size_s=300, sentence_timestamp=True)
    segments = normalize_segments_from_funasr(result)
    return {
        "raw_result": result,
        "segments": segments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe compressed demo video audio with FunASR")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--ingest-manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--wav-output", default=None)
    parser.add_argument("--model", default="paraformer-zh")
    parser.add_argument("--vad-model", default="fsmn-vad")
    parser.add_argument("--punc-model", default="ct-punc")
    parser.add_argument("--spk-model", default="cam++")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--keep-wav", action="store_true")
    args = parser.parse_args()

    case_v2_dir = ROOT / "data" / args.case_id / "v2"
    ingest_manifest_path = (
        Path(args.ingest_manifest) if args.ingest_manifest else (ROOT / "data" / args.case_id / "ingest_manifest.json")
    )
    output_path = Path(args.output) if args.output else (case_v2_dir / "asr" / "transcript_v1.json")
    wav_path = Path(args.wav_output) if args.wav_output else (case_v2_dir / "asr" / "audio_16k_mono.wav")

    ingest_manifest = read_json(ingest_manifest_path)
    video_path = resolve_video_path(ingest_manifest)
    run_ffmpeg_extract_wav(video_path, wav_path)
    transcribed = transcribe_with_funasr(
        wav_path=wav_path,
        model_name=args.model,
        vad_model=args.vad_model,
        punc_model=args.punc_model,
        spk_model=args.spk_model,
        device=args.device,
    )

    transcript = {
        "pipeline_stage": "criteria-trainer:asr:funasr",
        "generated_at": utc_now_iso(),
        "case_id": args.case_id,
        "source_video_path": str(video_path),
        "audio_wav_path": str(wav_path),
        "models": {
            "asr_model": args.model,
            "vad_model": args.vad_model,
            "punc_model": args.punc_model,
            "spk_model": args.spk_model,
            "device": args.device,
        },
        "segment_count": len(transcribed["segments"]),
        "segments": transcribed["segments"],
    }
    write_json(output_path, transcript)

    if not args.keep_wav and wav_path.exists():
        try:
            wav_path.unlink()
        except Exception:
            pass

    print(f"[OK] transcript: {output_path}")
    print(f"[OK] segments: {len(transcribed['segments'])}")


if __name__ == "__main__":
    main()

