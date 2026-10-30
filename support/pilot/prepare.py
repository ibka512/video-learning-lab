#!/usr/bin/env python3
"""Prepare and independently count a pilot's real media; standard library only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import wave


def dump(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def run(args: list[str]):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {args[0]}\n{result.stderr[-4000:]}")
    return result.stdout


def probe(path: Path, count=False):
    args = ["ffprobe", "-v", "error"]
    if count:
        args += ["-count_frames"]
    return json.loads(run(args+["-show_streams", "-show_format", "-of", "json", str(path)]))


def source_frames(path: Path):
    return json.loads(run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=pts,best_effort_timestamp,duration,pkt_duration,width,height,pict_type",
        "-of", "json", str(path)
    ]))["frames"]


def read_framehash(path: Path):
    tb, records = None, []
    for line in path.read_text().splitlines():
        match = re.match(r"#tb 0:\s*(\d+/\d+)", line)
        if match:
            tb = Fraction(match[1])
        if line.startswith("#") or not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            raise ValueError("invalid FFmpeg framehash record")
        records.append({"stream": int(parts[0]), "dts": int(parts[1]), "pts": int(parts[2]),
                        "duration": int(parts[3]), "bytes": int(parts[4]), "sha256_yuv": parts[5]})
    if not tb or not records:
        raise ValueError("missing time base or no decoded frames")
    return tb, records


def ffmpeg_new(args: list[str], destination: Path):
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-y", *args, str(temp)])
    temp.replace(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    work, out = args.workdir.resolve(), args.outdir.resolve()
    acquisition = json.loads((work / "acquisition.json").read_text())
    video, audio = work / "media/video.m4s", work / "media/audio.m4s"
    for item in acquisition["streams"]:
        path = Path(item["path"])
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise RuntimeError("acquired media changed: " + item["kind"])
    vp, ap = probe(video, count=True), probe(audio)
    vs, aus = vp["streams"][0], ap["streams"][0]
    tb = Fraction(vs["time_base"])
    origin = min(Fraction(vs["start_pts"]) * tb,
                 Fraction(aus["start_pts"]) * Fraction(aus["time_base"]))
    frames, counted = source_frames(video), int(vs["nb_read_frames"])
    if not frames or counted != len(frames):
        raise RuntimeError("source frame enumeration disagrees with complete decode count")
    hashfile = work / "video.framehash"
    if not hashfile.exists():
        temporary = hashfile.with_suffix(".tmp")
        run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-copyts", "-i", str(video),
             "-map", "0:v:0", "-an", "-fps_mode", "passthrough",
             "-enc_time_base", vs["time_base"], "-f", "framehash", "-hash", "sha256",
             "-y", str(temporary)])
        temporary.replace(hashfile)
    hash_tb, hashes = read_framehash(hashfile)
    if len(hashes) != counted:
        raise RuntimeError("FFmpeg framehash count differs from ffprobe count")
    (out / "sources").mkdir(parents=True, exist_ok=True)
    ledger_path = out / "sources/source-frames.jsonl.gz"
    ledger_temp = ledger_path.with_name(ledger_path.name + ".tmp")
    previous = None
    with gzip.open(ledger_temp, "wt", encoding="utf-8") as handle:
        for index, (frame, hashed) in enumerate(zip(frames, hashes, strict=True)):
            pts = frame.get("pts", frame.get("best_effort_timestamp"))
            if pts is None:
                raise RuntimeError(f"source PTS unavailable at frame {index}")
            point = int(pts) * tb
            if point != hashed["pts"] * hash_tb:
                raise RuntimeError(f"PTS mismatch between decode passes at frame {index}")
            if previous is not None and point < previous:
                raise RuntimeError("nonmonotonic presentation timestamp requires explicit handling")
            previous = point
            record = {"frame_id": f"F{index:07d}", "index": index, "pts": int(pts),
                      "time_base": str(tb), "part_time": float(point-origin),
                      "width": frame["width"], "height": frame["height"],
                      "pixel_format": vs["pix_fmt"], "sha256_yuv": hashed["sha256_yuv"],
                      "hash_decoder_time_base": str(hash_tb),
                      "media_version": acquisition["streams"][0]["sha256"],
                      "bvid": acquisition["bvid"], "cid": acquisition["cid"]}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    ledger_temp.replace(ledger_path)
    dump(work/"expected-native-frames.json",[
        {"frame_id":f"F{i:07d}","part_time":float(int(f["pts"])*tb-origin)}
        for i,f in enumerate(frames)])
    muxed = work / "media/source.mp4"
    ffmpeg_new(["-copyts", "-i", str(video), "-i", str(audio),
                "-map", "0:v:0", "-map", "1:a:0", "-c", "copy"], muxed)
    mux_video = next(s for s in probe(muxed, count=True)["streams"] if s["codec_type"] == "video")
    if int(mux_video["nb_read_frames"]) != counted:
        raise RuntimeError("remux changed the number of decoded frames")
    wav = work / "media/asr-16k.wav"
    ffmpeg_new(["-i", str(audio), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le"], wav)
    with wave.open(str(wav)) as handle:
        wav_samples, wav_rate = handle.getnframes(), handle.getframerate()
    indices = sorted({0, counted//2, counted-1})
    reference_dir = out / "evidence"
    reference_dir.mkdir(exist_ok=True)
    if not all((reference_dir/f"identity-{i+1:02d}.png").exists() for i in range(len(indices))):
        expression = "+".join(f"eq(n\\,{i})" for i in indices)
        run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(video),
             "-vf", "select="+expression, "-fps_mode", "passthrough", "-y",
             str(reference_dir/"identity-%02d.png")])
    references = [{"frame_id": f"F{idx:07d}", "index": idx,
                   "part_time": float(int(frames[idx]["pts"])*tb-origin),
                   "path": str(reference_dir/f"identity-{i+1:02d}.png")}
                  for i, idx in enumerate(indices)]
    manifest = {"schema_version": "0.1", "prepared_at": datetime.now(timezone.utc).isoformat(),
                "bvid": acquisition["bvid"], "cid": acquisition["cid"], "aid": acquisition["aid"],
                "title": acquisition["title"],
                "source_url": f'https://www.bilibili.com/video/{acquisition["bvid"]}/',
                "parts": acquisition["pages"], "source_video_sha256": sha256(video),
                "source_audio_sha256": sha256(audio), "source_frame_count": counted,
                "video_stream": vs, "audio_stream": aus, "common_origin": str(origin),
                "source_frame_ledger": str(ledger_path), "remuxed_media": str(muxed),
                "remuxed_sha256": sha256(muxed), "asr_audio": str(wav),
                "asr_audio_sha256": sha256(wav), "audio_sample_rate": wav_rate,
                "audio_samples": wav_samples, "audio_duration_decoded": wav_samples/wav_rate,
                "audio_start_in_part": float(Fraction(aus["start_pts"])*Fraction(aus["time_base"])-origin),
                "identity_evidence": references,
                "checks": {"source_hashes_match": True, "frame_count_matches": True,
                           "all_frame_pts_match_between_ffprobe_and_framehash": True,
                           "remux_frame_count_matches": True, "native_ocr_frame_mapping": "pending",
                           "content_identity_review": "pending"},
                "prepare_wall_seconds": time.monotonic()-start}
    dump(out/"sources/manifest.json", manifest)
    dump(out/"sources/coverage.json", {
        "schema_version": "0.1", "status": "prepared_not_inferred", "source_frames": counted,
        "decoded_frames_enumerated": len(frames), "independent_hashed_frames": len(hashes),
        "successful_vlm_unique_frames": 0, "semantic_input_coverage": 0.0,
        "asr_status": "not_started", "ocr_status": "not_started", "knowledge_status": "not_started",
        "issues": ["Local inference dependencies and model weights require confirmation."]})
    print(json.dumps({"manifest": str(out/"sources/manifest.json"), "frames": counted,
                      "video_duration": vs["duration"], "audio_duration": wav_samples/wav_rate,
                      "prepare_wall_seconds": manifest["prepare_wall_seconds"],
                      "references": references}, ensure_ascii=False))


if __name__ == "__main__":
    main()
