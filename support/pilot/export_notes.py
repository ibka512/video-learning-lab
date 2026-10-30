#!/usr/bin/env python3
"""Export evidence-linked Chinese Markdown from pilot knowledge records; no inference here."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from transcribe import timestamp

T=chr(96)
BASIS={"observed_visual":"画面直接可见","author_statement":"作者讲解",
       "observed_audio":"原音频","system_inference":"系统推断","externally_verified":"另行验证"}


def j(value):
    if isinstance(value,str):return value
    return json.dumps(value,ensure_ascii=False)


def cell(value):return j(value).replace("|","\\|").replace("\n"," ")


def read_lines(path):return list(map(json.loads,path.read_text().splitlines()))


def source_links(manifest,spans):
    return "；".join(f"[{timestamp(x['start'])}–{timestamp(x['end'])}]({manifest['source_url']}?p=1&t={int(x['start'])})" for x in spans)


def main():
    p=argparse.ArgumentParser();p.add_argument("--outdir",type=Path,required=True);a=p.parse_args()
    root=a.outdir.resolve();data=root/"sources"
    manifest=json.loads((data/"manifest.json").read_text())
    if manifest["bvid"]!="BV1cyHR67EbJ":
        raise ValueError("This export layout is specific to the Godot pilot; never apply its content to another video.")
    coverage=json.loads((data/"coverage.json").read_text())
    knowledge=read_lines(data/"knowledge.jsonl")
    events=read_lines(data/"events.jsonl")
    evidence_images={e["frame_id"]:root/"evidence"/Path(e["path"]).name for e in manifest["identity_evidence"]}
    for filename in ("detail-evidence.json","effect-evidence.json"):
        if (data/filename).exists():
            evidence_images.update({e["frame_id"]:root/"evidence"/Path(e["path"]).name for e in json.loads((data/filename).read_text())})
    evidence_images["F0000630"]=root/"evidence/setting-viewport-F0000630.png"
    def evidence_link(identity):
        if identity in evidence_images:return f"[{identity}](<{evidence_images[identity]}>)"
        if identity.startswith("SC"):return f"[{identity}](<{data/'transcript.corrected.md'}>)"
        return identity
    notes=root/"视频笔记";notes.mkdir(exist_ok=True)
    note=notes/"BV1cyHR67EbJ_窗口碎裂.md"
    strict=coverage.get("strict_input_complete",False)
    reviewed=coverage.get("knowledge_pipeline_complete",False)
    status="已完成本条处理与证据校订；本地小模型独立自动使用尚未通过" if strict and reviewed else "试运行草稿，尚未完成全片处理或复核"
    ratio=coverage.get("semantic_input_coverage")
    front={
        "schema_version":"0.1","video_id":manifest["bvid"],"cid":manifest["cid"],
        "title":manifest["title"],"author":"o中子脉冲o","source_url":manifest["source_url"],
        "published_at":manifest.get("published_at"),
        "processed_at":datetime.now(timezone.utc).isoformat() if strict and reviewed else None,
        "source_folders":["创作·设计","技术·工具"],"main_domain":"软件开发与游戏引擎",
        "tags":["Godot","Meta演出","透明窗口","2D动画"],"parts_total":1,
        "processing_profile":"strict_all_frames_temporal_patches",
        "semantic_input_coverage":ratio,"knowledge_units":len(knowledge),
        "quality_status":"codex_evidence_reviewed_local_vlm_not_approved" if reviewed else "draft"
    }
    lines=["---"]+[f"{k}: {json.dumps(v,ensure_ascii=False)}" for k,v in front.items()]+["---","",
           "# Godot 透明窗口与碎裂演出","",f"> {status}","",
           f"原视频：[{manifest['title']}]({manifest['source_url']}) · CID {manifest['cid']}","",
           "本条材料展示透明窗口的设置、嵌入式运行问题的处理，以及背景碎裂演出的制作方向。"
           "碎片生成、动画轨道和脚本实现没有展开，复现时需要补齐相应材料。","",
           "画面环境：Godot 4.7.2 stable，D3D12，NVIDIA GeForce RTX 3060 Laptop GPU。"
           "这些信息来自原片；本次没有在用户 Mac 上运行该 Godot 工程。","",
           "## 学习目标与前提","",
           "- 配合开启三个透明相关设置，并重启验证。",
           "- 识别嵌入式棋盘格窗口，找到取消运行时嵌入的方法。",
           "- 理解透明窗口、背景图与碎裂动画怎样构成演出。",
           "- 保留画面帮助提示中的性能、启动画面和平台差异。","",
           "## 完整材料","",
           f"- [带时间轴校订稿](<{data/'transcript.corrected.md'}>)",
           f"- [ASR 原稿与时间轴](<{data/'transcript.md'}>)",
           f"- [校订 VTT](<{data/'transcript.corrected.vtt'}>)",
           f"- [校订 SRT](<{data/'transcript.corrected.srt'}>)",
           f"- [修订记录](<{data/'transcript-corrections.json'}>)",
           f"- [结构化知识](<{data/'knowledge.jsonl'}>)",
           f"- [事件时间轴](<{data/'events.jsonl'}>)",
           f"- [覆盖报告](<{data/'coverage.json'}>)","",
           "## 时间导航","",
           "| 时间范围 | 内容 | 知识 |","|---|---|---|"]
    for e in events:
        labels="、".join(x.rsplit("/",1)[-1] for x in e["knowledge_ids"]) or e.get("no_new_knowledge_reason","")
        lines.append(f"| {timestamp(e['start'])}–{timestamp(e['end'])} | {cell(e['title'])} | {cell(labels)} |")
    lines += ["","## 三个设置的速查","",
              "| 画面路径 | 精确设置名 | 视频要求 |","|---|---|---|",
              f"| 显示 → 窗口 → 大小 → 透明 | {T}display/window/size/transparent{T} | true |",
              f"| 显示 → 窗口 → 像素级透明 → 允许 | {T}display/window/per_pixel_transparency/allowed{T} | true |",
              f"| 渲染 → 视口 → 透明背景 | {T}rendering/viewport/transparent_background{T} | true |","",
              "设置目标依据作者讲解与当前勾选状态。帮助提示中的 bool = false 是另一条显示信息，"
              "不能据此把本视频的开启步骤写成关闭。",""]
    for k in knowledge:
        lines += [f"## {k['short_id']} · {k['title']}","",
                  f"**结论：** {k['claim']['text']}","",
                  f"**依据：** {BASIS.get(k['claim']['basis'],k['claim']['basis'])} · {source_links(manifest,k['source_spans'])}",""]
        if k["prerequisites"]:
            lines += ["**前提：** "+"；".join(k["prerequisites"]),""] 
        if k["steps"]:
            lines += ["| 步骤 | 动作 | 参数/结果 | 依据 |","|---|---|---|---|"]
            for i,s in enumerate(k["steps"]):
                details=[]
                if s.get("parameters"):details.append(j(s["parameters"]))
                if s.get("expected_result"):details.append(s["expected_result"])
                ev=s.get("evidence_frames",[])+s.get("evidence_transcript",[])
                lines.append(f"| {i+1} | {cell(s['action'])} | {cell('；'.join(details))} | {cell(BASIS.get(s.get('basis'),s.get('basis',''))+'：'+', '.join(ev))} |")
            lines.append("")
        if k["parameters"]:
            lines += ["| 参数/内容 | 已确认值 |","|---|---|"]
            for item in k["parameters"]:lines.append(f"| {cell(item['name'])} | {cell(item['value'])} |")
            lines.append("")
        for exp in k["experience"]:
            lines += ["**经验：** "+"；".join(f"{label}：{j(exp[key])}" for key,label in
                      [("scenario","情境"),("symptom","现象"),("reason","依据/原因"),
                       ("adjustment","调整"),("result","结果")] if key in exp),""]
        if k["limitations"]:lines += ["**边界与缺失：** "+"；".join(k["limitations"]),""] 
        lines += [f"**证据 ID：** {', '.join(evidence_link(identity) for identity in k['evidence_frames']+k['evidence_transcript'])}","",
                  f"**审核状态：** {k['review_status']}",""]
    lines += ["## 关键证据图",""]
    figures=[
        ("显示/窗口/大小/透明",15.0,"F0000450",root/"evidence/detail-03.png"),
        ("逐像素透明许可",19.0,"F0000570",root/"evidence/detail-04.png"),
        ("根视口透明背景",21.0,"F0000630",root/"evidence/setting-viewport-F0000630.png"),
        ("透明结果与灰底注释",25.0,"F0000750",root/"evidence/detail-05.png"),
        ("运行时嵌入选项",34.4,"F0001032",root/"evidence/identity-02.png"),
        ("碎裂后显露文字与手部图片",54.0,"F0001620",root/"evidence/effect-02.png"),
        ("不同形状的背景变化",59.5,"F0001785",root/"evidence/effect-03.png")]
    for label,t,fid,path in figures:
        if path.exists():
            lines += [f"**{label} · {timestamp(t)} · {fid}**","",f"![{label}](<{path}>)",""]
    lines += ["## AI 使用与未验证事项","",
              "- 回答操作问题时一并给出三项设置、重启要求和嵌入式运行条件。",
              "- 将作者讲解、画面帮助提示和系统推断分别归属来源。",
              "- 跨平台或跨版本使用前检查适用条件；原片没有验证用户的 Mac 环境。",
              "- 碎片算法、脚本、动画轨道、时长和数值是未展示内容，使用时明确这些缺口。",
              "- 转写时间为强制对齐模型估计；未报告人工 CER/WER 和时间边界误差达标。",
              "- 非语音声音没有独立听辨，不能从空转写区间推断其必然静音。",""]
    note.write_text("\n".join(lines))
    index=root/"领域索引";index.mkdir(exist_ok=True)
    domain=["# 软件开发与游戏引擎","",
            f"当前首条试运行来源：[Godot 透明窗口与碎裂演出](<{note}>)。","",
            "| AI 要解决的问题 | 知识 ID | 条件/证据 |","|---|---|---|"]
    for k in knowledge:
        domain.append(f"| {cell(k['title'])} | {k['short_id']} | {source_links(manifest,k['source_spans'])} |")
    (index/"软件开发与游戏引擎.md").write_text("\n".join(domain)+"\n")
    overview=["# 首条视频真实验证","",f"状态：{status}。","",
              f"- [视频知识笔记](<{note}>)",
              f"- [带时间轴校订转写](<{data/'transcript.corrected.md'}>)",
              f"- [领域索引](<{index/'软件开发与游戏引擎.md'}>)",
              f"- [环境与依赖清单](<{root/'环境检查与依赖清单.md'}>)",
              f"- [工具和运行说明](<{root/'工具/README.md'}>)",
              f"- [结构化覆盖报告](<{data/'coverage.json'}>)",
              f"- [AI读取说明](<{root/'AI读取说明.md'}>)",
              f"- [试运行报告](<{root/'试运行报告.md'}>)","",
              "本目录针对一条约 69 秒的实际视频。此前 739 条视频的元数据预整理保持原样；"
              "本目录不代表其余收藏已完成声画分析。","",
              "环境、下载缓存、模型和原始媒体均位于外接固态盘当前项目的 work/pilot-20261026 中。"]
    (root/"README.md").write_text("\n".join(overview)+"\n")
    print(json.dumps({"note":str(note),"knowledge_units":len(knowledge),"status":status},ensure_ascii=False))


if __name__=="__main__":
    main()
