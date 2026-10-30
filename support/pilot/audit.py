#!/usr/bin/env python3
"""Audit source IDs, OCR timestamps, and actual VLM input receipts independently of progress counters."""
from __future__ import annotations
import argparse,gzip,json,math
from fractions import Fraction
from pathlib import Path
from prepare import dump,read_framehash


def check_frame_coverage(source,requests):
    expected={item["frame_id"]:item for item in source}
    if len(expected)!=len(source):raise ValueError("duplicate IDs in source enumeration")
    covered=set();failed=[];problems=[]
    for item in requests:
        if item.get("status")!="success":
            failed.append(item.get("window_index"));continue
        ids=item["frame_ids"]
        if len(ids)!=len(set(ids)) or not set(ids)<=set(expected):
            problems.append("request has duplicate or unknown source IDs");continue
        inputs=item["source_inputs"]
        if [x["frame_id"] for x in inputs]!=ids:
            problems.append("source input order does not match submitted IDs");continue
        if any(abs(x["time"]-expected[x["frame_id"]]["part_time"])>1e-8 for x in inputs):
            problems.append("request source times do not match original PTS");continue
        tensor=item["preprocessor"]
        if tensor.get("representation")=="temporal_video_patches":
            grid=tensor["video_grid_thw"]
            padding=tensor["padding_frames"];temporal=tensor["temporal_patch_size"]
            packed_ids=[f for group in tensor["temporal_groups"] for f in group["frame_ids"]]
            wanted_ids=ids+[ids[-1]]*padding
            if (tensor["source_frame_count"]!=len(ids) or not 0<=padding<temporal or
                packed_ids!=wanted_ids or len(grid)!=1 or grid[0][0]*temporal!=len(ids)+padding):
                problems.append("temporal packing lost or reordered source frames");continue
        else:
            grid=tensor["image_grid_thw"]
            if len(grid)!=len(ids) or any(len(row)!=3 or row[0]!=1 for row in grid):
                problems.append("not all source images survived preprocessing");continue
        patches=sum(math.prod(row) for row in grid)
        if tensor["pixel_values_shape"][0]!=patches:
            problems.append("pixel tensor size does not match image grids");continue
        actual_tokens=tensor.get("media_pad_token_count",tensor.get("image_pad_token_count"))
        wanted_tokens=tensor.get("expected_media_tokens",tensor.get("expected_image_tokens"))
        if actual_tokens!=wanted_tokens:
            problems.append("image token markers are incomplete");continue
        if not item.get("observation"):
            problems.append("no validated model output");continue
        covered.update(ids)
    missing=sorted(set(expected)-covered)
    return {"source_frames":len(source),"successful_unique_frames":len(covered),
            "semantic_input_coverage":len(covered)/len(source) if source else None,
            "missing_frame_ids":missing,"failed_windows":failed,"receipt_problems":problems,
            "strict_input_complete":bool(source) and not missing and not problems}


def main():
    p=argparse.ArgumentParser();p.add_argument("--outdir",type=Path,required=True);args=p.parse_args()
    root=args.outdir.resolve();data=root/"sources"
    manifest=json.loads((data/"manifest.json").read_text())
    with gzip.open(data/"source-frames.jsonl.gz","rt") as handle:source=list(map(json.loads,handle))
    hash_tb,hash_rows=read_framehash(data/"source-video.framehash")
    independent_hash_complete=len(hash_rows)==len(source)
    if independent_hash_complete:
        independent_hash_complete=all(
            Fraction(x["pts"])*Fraction(x["time_base"])==row["pts"]*hash_tb and
            x["sha256_yuv"]==row["sha256_yuv"]
            for x,row in zip(source,hash_rows,strict=True))
    requests=[json.loads(p.read_text()) for p in sorted((data/"inputs").glob("request-*.json"))] if (data/"inputs").exists() else []
    result=check_frame_coverage(source,requests)
    if not independent_hash_complete:result["strict_input_complete"]=False
    ocr_path=data/"ocr.jsonl"
    ocr=list(map(json.loads,ocr_path.read_text().splitlines())) if ocr_path.exists() else []
    ocr_ok=len(ocr)==len(source)
    if ocr_ok:
        for actual,expected in zip(ocr,source,strict=True):
            actual_time=Fraction(actual["native_pts"])*Fraction(actual["native_time_base"])-Fraction(manifest["common_origin"])
            source_time=Fraction(expected["pts"])*Fraction(expected["time_base"])-Fraction(manifest["common_origin"])
            if actual["frame_id"]!=expected["frame_id"] or abs(actual_time-source_time)>Fraction(1,16000):
                ocr_ok=False;break
    speech=(data/"asr-metrics.json").exists() and (data/"words.jsonl").exists()
    events=list(map(json.loads,(data/"events.jsonl").read_text().splitlines())) if (data/"events.jsonl").exists() else []
    knowledge=list(map(json.loads,(data/"knowledge.jsonl").read_text().splitlines())) if (data/"knowledge.jsonl").exists() else []
    expected_frame_ids={x["frame_id"] for x in source}
    event_frame_ids={f for e in events for f in e.get("frame_ids",[])}
    event_frames_complete=event_frame_ids==expected_frame_ids
    cursor=0.0;event_time_complete=bool(events)
    for event in sorted(events,key=lambda e:e["start"]):
        if event["start"]>cursor+1e-6 or event["end"]<event["start"]:
            event_time_complete=False
        cursor=max(cursor,event["end"])
    if cursor+1e-6<manifest["audio_start_in_part"]+manifest["audio_duration_decoded"]:
        event_time_complete=False
    mapped=(all(e.get("knowledge_ids") or e.get("no_new_knowledge_reason") for e in events)
            and event_frames_complete and event_time_complete) if events else False
    second_pass=all(e.get("second_pass_reviewed") is True for e in events) if events else False
    ids={item["frame_id"] for item in source}
    seg_path=data/"transcript.segments.jsonl"
    segment_ids={x["segment_id"] for x in map(json.loads,seg_path.read_text().splitlines())} if seg_path.exists() else set()
    corrected_path=data/"transcript.corrected.segments.jsonl"
    if corrected_path.exists():segment_ids.update(x["segment_id"] for x in map(json.loads,corrected_path.read_text().splitlines()))
    evidence_problems=[]
    known_ids={k["knowledge_id"] for k in knowledge}
    for k in knowledge:
        if not k.get("source_spans"):evidence_problems.append(k["knowledge_id"]+": missing time span")
        for f in k.get("evidence_frames",[]):
            if f not in ids:evidence_problems.append(k["knowledge_id"]+": unknown frame "+f)
        for s in k.get("evidence_transcript",[]):
            if s not in segment_ids:evidence_problems.append(k["knowledge_id"]+": unknown transcript "+s)
        if not k.get("evidence_frames") and not k.get("evidence_transcript"):
            evidence_problems.append(k["knowledge_id"]+": no evidence")
    for event in events:
        if not set(event.get("knowledge_ids",[]))<=known_ids:evidence_problems.append("event references absent knowledge")
    knowledge_reviewed=bool(knowledge) and all(k.get("review_status")=="codex_evidence_reviewed" for k in knowledge)
    notes_ready=knowledge_reviewed and bool(events) and mapped and second_pass and not evidence_problems
    result.update({"schema_version":"0.1","bvid":manifest["bvid"],"cid":manifest["cid"],
                   "decoded_frames_enumerated":len(source),"independent_hashed_frames":len(hash_rows),
                   "independent_hash_and_pts_match":independent_hash_complete,
                   "ocr_frames":len(ocr),"ocr_pts_mapping_complete":ocr_ok,
                   "speech_pipeline_complete":speech,"events":len(events),
                   "event_mapping_complete":mapped,"events_second_pass_complete":second_pass,
                   "event_frame_partition_complete":event_frames_complete,
                   "event_time_coverage_complete":event_time_complete,
                   "knowledge_review_complete":knowledge_reviewed,
                   "knowledge_units":len(knowledge),"knowledge_evidence_problems":evidence_problems,
                   "knowledge_pipeline_complete":notes_ready,
                   "human_CER_WER_time_boundary_measurement":"not_performed",
                   "input_coverage_is_understanding_accuracy":False})
    assessment_path=data/"quality-assessment.json"
    if assessment_path.exists():
        assessment=json.loads(assessment_path.read_text())
        result["local_vlm_unattended_use_approved"]=assessment.get("local_vlm_unattended_use_approved",False)
    done=result["strict_input_complete"] and ocr_ok and speech and notes_ready and not result["failed_windows"]
    result["status"]="pilot_processed_with_source_corrections_model_quality_limited" if done else "in_progress"
    dump(data/"coverage.json",result)
    print(json.dumps({**{k:v for k,v in result.items() if k!="missing_frame_ids"},"missing_frame_count":len(result["missing_frame_ids"])},ensure_ascii=False))


if __name__=="__main__":
    main()
