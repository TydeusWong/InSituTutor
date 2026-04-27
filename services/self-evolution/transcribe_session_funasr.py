import argparse
import sys
from pathlib import Path

from common import read_json, rel_path, session_root, utc_now_iso, write_json


ROOT = Path(__file__).resolve().parents[2]
CRITERIA_TRAINER = ROOT / "services" / "criteria-trainer"
if str(CRITERIA_TRAINER) not in sys.path:
    sys.path.insert(0, str(CRITERIA_TRAINER))

from transcribe_audio_funasr import normalize_segments_from_funasr, run_ffmpeg_extract_wav, transcribe_with_funasr  # noqa: E402


def resolve_cached_model(model_id: str) -> str:
    if not model_id or Path(model_id).exists():
        return model_id
    aliases = {
        "paraformer-zh": "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "fsmn-vad": "speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "ct-punc": "punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        "cam++": "speech_campplus_sv_zh-cn_16k-common",
    }
    leaf = aliases.get(model_id, model_id)
    base = Path.home() / ".cache" / "modelscope" / "hub" / "models" / "iic"
    candidate = base / leaf
    return str(candidate) if candidate.exists() else model_id


def normalize_turns(segments: list[dict]) -> list[dict]:
    turns: list[dict] = []
    for idx, seg in enumerate(segments, start=1):
        speaker = str(seg.get("speaker") or seg.get("speaker_id") or "unknown")
        turns.append(
            {
                "turn_id": f"turn_{idx:04d}",
                "speaker_id": speaker,
                "start_sec": float(seg.get("start_sec", 0.0) or 0.0),
                "end_sec": float(seg.get("end_sec", 0.0) or 0.0),
                "text": str(seg.get("text", "")).strip(),
            }
        )
    return turns


def transcribe_session(
    *,
    case_id: str,
    session_id: str,
    device: str,
    model: str,
    vad_model: str,
    punc_model: str,
    spk_model: str,
    allow_empty: bool,
) -> tuple[Path, Path]:
    base = session_root(case_id, session_id)
    manifest_path = base / "session_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    raw_dir = base / "raw"
    audio_source = Path(str(manifest.get("raw_audio_path") or raw_dir / "teaching_audio.webm"))
    if not audio_source.is_absolute():
        audio_source = ROOT / audio_source
    review_media = Path(str(manifest.get("review_media_path") or raw_dir / "teaching_session_review.mp4"))
    if not review_media.is_absolute():
        review_media = ROOT / review_media
    if not audio_source.exists() and review_media.exists():
        audio_source = review_media
    if not audio_source.exists():
        raise FileNotFoundError(f"missing session audio: {audio_source}")

    wav_path = raw_dir / "audio.wav"
    run_ffmpeg_extract_wav(audio_source, wav_path)
    segments: list[dict]
    raw_result = None
    try:
        transcribed = transcribe_with_funasr(
            wav_path=wav_path,
            model_name=resolve_cached_model(model),
            vad_model=resolve_cached_model(vad_model),
            punc_model=resolve_cached_model(punc_model),
            spk_model=resolve_cached_model(spk_model),
            device=device,
        )
        raw_result = transcribed.get("raw_result")
        segments = transcribed.get("segments", [])
    except Exception as exc:
        if not allow_empty:
            raise
        segments = []
        raw_result = {"error": str(exc), "fallback": "empty_transcript"}

    turns = normalize_turns(segments)
    if not spk_model:
        for turn in turns:
            turn["speaker_id"] = "unknown"
    transcript = {
        "pipeline_stage": "self-evolution:asr:funasr",
        "generated_at": utc_now_iso(),
        "case_id": case_id,
        "session_id": session_id,
        "source_audio_path": rel_path(audio_source),
        "audio_wav_path": rel_path(wav_path),
        "models": {
            "asr_model": model,
            "vad_model": vad_model,
            "punc_model": punc_model,
            "spk_model": spk_model,
            "device": device,
        },
        "segment_count": len(turns),
        "segments": turns,
        "raw_result": raw_result,
    }
    transcript_path = base / "asr" / "transcript_v1.json"
    speaker_turns_path = base / "asr" / "speaker_turns_v1.json"
    write_json(transcript_path, transcript)
    write_json(
        speaker_turns_path,
        {
            "pipeline_stage": "self-evolution:asr:speaker-turns",
            "generated_at": utc_now_iso(),
            "case_id": case_id,
            "session_id": session_id,
            "turn_count": len(turns),
            "speaker_turns": turns,
        },
    )
    return transcript_path, speaker_turns_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a self-evolution session with FunASR.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="paraformer-zh")
    parser.add_argument("--vad-model", default="fsmn-vad")
    parser.add_argument("--punc-model", default="ct-punc")
    parser.add_argument(
        "--spk-model",
        default="",
        help="Optional speaker model. Leave empty for fast ASR; use cam++ when diarization is required.",
    )
    parser.add_argument("--allow-empty", action="store_true", help="Write empty transcript if FunASR is unavailable.")
    args = parser.parse_args()
    transcript_path, turns_path = transcribe_session(
        case_id=args.case_id,
        session_id=args.session_id,
        device=args.device,
        model=args.model,
        vad_model=args.vad_model,
        punc_model=args.punc_model,
        spk_model=args.spk_model,
        allow_empty=args.allow_empty,
    )
    print(f"[OK] transcript: {transcript_path}")
    print(f"[OK] speaker turns: {turns_path}")


if __name__ == "__main__":
    main()
