#!/usr/bin/env python3
"""Pilot speech recognition and independent forced alignment, strictly local model paths."""
from __future__ import annotations
import argparse
from dataclasses import asdict, is_dataclass
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import re
import time
import unicodedata
from prepare import dump, sha256


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def restore_transcript(text, units):
    """Map normalized aligner units to source text without guessing missing characters."""
    mapped = []
    letters = []
    for i, c in enumerate(text):
        for n in normalized(c):
            letters.append(n)
            mapped.append(i)
    unit_text = "".join(normalized(u["text"]) for u in units)
    if unit_text != "".join(letters):
        raise ValueError("aligner units do not account for the original transcript text")
    positions, cursor = [], 0
    for item in units:
        token = normalized(item["text"])
        if not token:
            continue
        positions.append((mapped[cursor], item))
        cursor += len(token)
    if not positions:
        raise ValueError("no aligned units")
    restored = []
    for i, (begin, item) in enumerate(positions):
        begin = 0 if i == 0 else begin
        end = positions[i+1][0] if i+1 < len(positions) else len(text)
        restored.append({**item, "aligned_token": item["text"], "text": text[begin:end]})
    if "".join(u["text"] for u in restored) != text:
        raise ValueError("restoring punctuation lost transcript content")
    return restored


def timestamp(seconds, separator="."):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("invalid subtitle time")
    milliseconds = round(seconds*1000)
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{milliseconds:03}"


def sentences(words):
    result, group = [], []
    for unit in words:
        group.append(unit)
        span = unit["end"]-group[0]["start"]
        ends = bool(re.search(r"[。！？!?；;\n]\s*$", unit["text"]))
        phrase_end = bool(re.search(r"[，,：:]\s*$", unit["text"]))
        if ends or (phrase_end and span >= 2.5) or span >= 12:
            result.append({"segment_id": f"S{len(result):04d}",
                           "start": group[0]["start"], "end": group[-1]["end"],
                           "text": "".join(x["text"] for x in group),
                           "word_ids": [x["word_id"] for x in group]})
            group = []
    if group:
        result.append({"segment_id": f"S{len(result):04d}",
                       "start": group[0]["start"], "end": group[-1]["end"],
                       "text": "".join(x["text"] for x in group),
                       "word_ids": [x["word_id"] for x in group]})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()
    out, model_root = args.outdir.resolve(), args.model_root.resolve()
    manifest = json.loads((out/"sources/manifest.json").read_text())
    audio = Path(manifest["asr_audio"])
    if sha256(audio) != manifest["asr_audio_sha256"]:
        raise RuntimeError("audio content changed")
    if manifest["audio_duration_decoded"] > 180:
        raise RuntimeError("pilot ASR requires a <=180s input; long-file chunking not implemented here")
    # Set paths before importing any Hugging Face or MLX module.
    import os
    cache = model_root.parent/"cache/huggingface"
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import mlx.core as mx
    from mlx_audio.stt.utils import load_model
    asr_path = model_root/"Qwen3-ASR-1.7B-8bit"
    align_path = model_root/"Qwen3-ForcedAligner-0.6B-8bit"
    asr_receipt = json.loads((asr_path/"pilot-download-receipt.json").read_text())
    align_receipt = json.loads((align_path/"pilot-download-receipt.json").read_text())
    identity = {
        "audio_sha256": manifest["asr_audio_sha256"],
        "asr_revision": asr_receipt["revision"], "aligner_revision": align_receipt["revision"],
        "mlx_audio": importlib.metadata.version("mlx-audio"), "mlx": importlib.metadata.version("mlx"),
        "language": "Chinese", "schema_version": "0.1", "prompt_version": "asr-independent-v1"
    }
    identity_path = out/"sources/asr-identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise RuntimeError("ASR checkpoint identity changed; preserve old results and use another output")
    dump(identity_path, identity)
    started = time.monotonic()
    raw_path = out/"sources/transcript.raw.json"
    if raw_path.exists():
        raw = json.loads(raw_path.read_text())
    else:
        loaded = time.monotonic()
        model = load_model(str(asr_path))
        print(json.dumps({"event":"asr_model_loaded","seconds":time.monotonic()-loaded}), flush=True)
        result = model.generate(str(audio), language="Chinese", temperature=0, max_tokens=4096,
                                chunk_duration=180, verbose=False)
        raw = asdict(result) if is_dataclass(result) else dict(vars(result))
        raw["source_audio_sha256"] = manifest["asr_audio_sha256"]
        raw["part_start"] = manifest["audio_start_in_part"]
        raw["part_end"] = raw["part_start"]+manifest["audio_duration_decoded"]
        raw["wall_seconds"] = time.monotonic()-loaded
        raw["peak_memory_gb"] = mx.get_peak_memory()/1e9
        if not raw.get("text", "").strip():
            raise RuntimeError("empty ASR output requires review")
        if raw.get("generation_tokens", 0) >= 4096:
            raise RuntimeError("ASR token budget reached; do not accept truncated transcript")
        dump(raw_path, raw)
        print(json.dumps({"event":"asr_complete","text":raw["text"],
                          "seconds":raw["wall_seconds"],"peak_memory_gb":raw["peak_memory_gb"]},
                         ensure_ascii=False), flush=True)
        del result, model
        gc.collect()
        mx.clear_cache()
    aligned_path = out/"sources/aligner.raw.json"
    if aligned_path.exists():
        aligned = json.loads(aligned_path.read_text())
    else:
        loaded = time.monotonic()
        aligner = load_model(str(align_path))
        result = aligner.generate(str(audio), text=raw["text"], language="Chinese")
        aligned = asdict(result) if is_dataclass(result) else dict(vars(result))
        aligned["wall_seconds"] = time.monotonic()-loaded
        aligned["peak_memory_gb"] = mx.get_peak_memory()/1e9
        dump(aligned_path, aligned)
        del result, aligner
        gc.collect()
        mx.clear_cache()
    duration = manifest["audio_duration_decoded"]
    shift = manifest["audio_start_in_part"]
    units, previous = [], -1
    for index, item in enumerate(aligned["items"]):
        start, end = item["start_time"], item["end_time"]
        if not (math.isfinite(start) and math.isfinite(end) and
                0 <= start <= end <= duration+0.02 and start >= previous):
            raise RuntimeError("forced alignment is out of bounds or nonmonotonic")
        previous = start
        units.append({"word_id":f"W{index:04d}", "text":item["text"],
                      "start":round(start+shift,6), "end":round(end+shift,6)})
    restored = restore_transcript(raw["text"], units)
    segments = sentences(restored)
    for filename, items in [("words.jsonl",restored),("transcript.segments.jsonl",segments)]:
        temporary = out/"sources"/(filename+".tmp")
        temporary.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in items))
        temporary.replace(out/"sources"/filename)
    vtt = ["WEBVTT\n"]
    srt = []
    md = [f"# {manifest['title']}：独立音频转写", "",
          "> 状态：ASR 原稿经强制对齐；尚未人工逐句校订。", "",
          f"来源：{manifest['source_url']} · CID {manifest['cid']}", ""]
    for i, seg in enumerate(segments):
        vtt += [f"{timestamp(seg['start'])} --> {timestamp(seg['end'])}",seg["text"],""]
        srt += [str(i+1),f"{timestamp(seg['start'],',')} --> {timestamp(seg['end'],',')}",seg["text"],""]
        md += [f"**{timestamp(seg['start'])}–{timestamp(seg['end'])} · {seg['segment_id']}**", "",
               seg["text"], ""]
    (out/"sources/transcript.vtt").write_text("\n".join(vtt))
    (out/"sources/transcript.srt").write_text("\n".join(srt))
    (out/"sources/transcript.md").write_text("\n".join(md))
    gaps, cursor = [], shift
    for item in restored:
        if item["start"] > cursor+0.25:
            gaps.append({"start":cursor,"end":item["start"],"classification":"untranscribed_gap",
                         "review":"pending; not automatically assumed silent"})
        cursor = max(cursor,item["end"])
    if cursor < shift+duration-0.25:
        gaps.append({"start":cursor,"end":shift+duration,"classification":"untranscribed_gap",
                     "review":"pending"})
    dump(out/"sources/audio-timeline.json", {
        "input_span":{"start":shift,"end":shift+duration,"status":"entire_audio_submitted_to_asr"},
        "alignment_text_accounted":True, "untranscribed_gaps":gaps,
        "non_speech_semantics":"not inferred by ASR; requires source review"})
    metrics = {"wall_seconds_this_run":time.monotonic()-started, "asr_wall_seconds":raw["wall_seconds"],
               "aligner_wall_seconds":aligned["wall_seconds"],
               "audio_seconds":duration, "asr_realtime_factor":raw["wall_seconds"]/duration,
               "segments":len(segments), "aligned_units":len(units),
               "text_characters":len(raw["text"]), "punctuation_restored_without_text_loss":True,
               "human_quality_measurement":"not_performed"}
    dump(out/"sources/asr-metrics.json", metrics)
    print(json.dumps({"event":"speech_pipeline_complete",**metrics},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
