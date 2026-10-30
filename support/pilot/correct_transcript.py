#!/usr/bin/env python3
"""Align a separately reviewed correction without overwriting raw ASR evidence."""
from pathlib import Path
import argparse,json,os,time
from dataclasses import asdict
from transcribe import restore_transcript,sentences,timestamp
from prepare import dump,sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--outdir",type=Path,required=True)
    p.add_argument("--aligner",type=Path,required=True)
    p.add_argument("--text",type=Path,required=True)
    args=p.parse_args()
    out=args.outdir.resolve();data=out/"sources"
    manifest=json.loads((data/"manifest.json").read_text())
    text=args.text.read_text().strip()
    os.environ["HF_HOME"]=str(args.aligner.resolve().parent.parent/"cache/huggingface")
    os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
    import mlx.core as mx
    from mlx_audio.stt.utils import load_model
    identity={"audio_sha256":manifest["asr_audio_sha256"],"text_sha256":sha256(args.text),
              "aligner_revision":json.loads((args.aligner/"pilot-download-receipt.json").read_text())["revision"]}
    started=time.monotonic()
    model=load_model(str(args.aligner))
    result=model.generate(manifest["asr_audio"],text=text,language="Chinese")
    raw=asdict(result)
    raw["identity"]=identity
    raw["wall_seconds"]=time.monotonic()-started
    raw["peak_memory_gb"]=mx.get_peak_memory()/1e9
    dump(data/"aligner.corrected.json",raw)
    duration=manifest["audio_duration_decoded"];shift=manifest["audio_start_in_part"]
    units=[];previous=-1
    for i,item in enumerate(raw["items"]):
        a,b=item["start_time"],item["end_time"]
        if not 0<=a<=b<=duration+0.02 or a<previous:raise ValueError("corrected alignment out of bounds")
        previous=a
        units.append({"word_id":f"WC{i:04d}","text":item["text"],"start":round(a+shift,6),"end":round(b+shift,6)})
    restored=restore_transcript(text,units)
    segments=sentences(restored)
    for i,s in enumerate(segments):s["segment_id"]=f"SC{i:04d}"
    for name,items in [("words.corrected.jsonl",restored),("transcript.corrected.segments.jsonl",segments)]:
        (data/name).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in items))
    md=[f"# {manifest['title']}：带时间轴校订稿","",
        "> 依据原片界面术语与画面内字幕校订，重新做强制对齐；未进行人工逐句听辨和时间误差测量。","",
        f"来源：{manifest['source_url']} · CID {manifest['cid']}","",
        "原稿：transcript.raw.json；逐项修订：transcript-corrections.json。",""]
    vtt=["WEBVTT\n"];srt=[]
    for i,s in enumerate(segments):
        md += [f"**{timestamp(s['start'])}–{timestamp(s['end'])} · {s['segment_id']}**","",s["text"],""]
        vtt += [f"{timestamp(s['start'])} --> {timestamp(s['end'])}",s["text"],""]
        srt += [str(i+1),f"{timestamp(s['start'],',')} --> {timestamp(s['end'],',')}",s["text"],""]
    (data/"transcript.corrected.md").write_text("\n".join(md))
    (data/"transcript.corrected.vtt").write_text("\n".join(vtt))
    (data/"transcript.corrected.srt").write_text("\n".join(srt))
    print(json.dumps({"event":"corrected_alignment_complete","segments":len(segments),
                      "aligned_units":len(units),"wall_seconds":raw["wall_seconds"],
                      "peak_memory_gb":raw["peak_memory_gb"]},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
