#!/usr/bin/env python3
"""Build an honest pilot report from measured artifacts, keeping coverage and accuracy separate."""
import argparse,json
from pathlib import Path
from transcribe import timestamp


def load(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    p=argparse.ArgumentParser();p.add_argument("--outdir",type=Path,required=True);a=p.parse_args()
    root=a.outdir.resolve();d=root/"sources"
    m=load(d/"manifest.json");c=load(d/"coverage.json")
    if m["bvid"]!="BV1cyHR67EbJ":
        raise ValueError("This report template is specific to the measured Godot pilot.")
    speech=load(d/"asr-metrics.json",{});corrected=load(d/"aligner.corrected.json",{})
    ocr=load(d/"ocr-metrics.json",{});vision=load(d/"vision-metrics.json",{})
    quality=load(d/"quality-assessment.json",{})
    requests=[load(x) for x in sorted((d/"inputs").glob("request-*.json"))]
    success=[x for x in requests if x["status"]=="success"]
    source_occurrences=sum(len(x["frame_ids"]) for x in success)
    pad=sum(x["preprocessor"].get("padding_frames",0) for x in success)
    shapes=sorted({(x["preprocessor"].get("resized_frame_width"),
                   x["preprocessor"].get("resized_frame_height")) for x in success})
    finished=c.get("strict_input_complete") and c.get("knowledge_pipeline_complete")
    lines=["# 首条视频试运行报告","",
           ("结果：首条实际视频的全帧处理、时间轴转写与知识证据校订已完成。" if finished else "结果：首条视频仍在处理；最终完成状态以覆盖报告为准。"),"",
           "**质量判断：本地 4B 视觉模型存在已观察到的语义误判，尚未批准无人复核使用。"
           "结构化输出解决格式问题，不能解决所有理解问题。最终知识需按原片证据校订。**","",
           "## 样本与范围","",
           f"- [{m['title']}]({m['source_url']})",
           f"- BV {m['bvid']}，CID {m['cid']}；实际分 P 数为 {len(m['parts'])}。",
           f"- 1920 × 1080，30 FPS，视频 {m['video_stream']['duration']} 秒；原音轨约 68.842667 秒。",
           "- 本轮针对这一条视频，其余收藏与 12 类代表样本未启动。",
           "- 前轮基于元数据的 739 条预整理未改写。","",
           "## 完整性与质量分别报告","",
           "| 项目 | 实际结果 |","|---|---|",
           f"| 源帧枚举 | {c['source_frames']} 帧 |",
           f"| 第二次解码像素 hash 与 PTS | {c['independent_hashed_frames']} 帧；匹配：{c.get('independent_hash_and_pts_match')} |",
           f"| Apple 原生 OCR | {c['ocr_frames']} 帧；源时间映射完整：{c['ocr_pts_mapping_complete']} |",
           f"| 成功视觉输入唯一源帧 | {c['successful_unique_frames']} / {c['source_frames']} |",
           f"| 严格输入覆盖 | {c['semantic_input_coverage']:.2%} |",
           f"| 当前配置失败窗口 | {len(c['failed_windows'])} |",
           f"| 声画事件 | {c['events']}；源帧与全时段覆盖：{c.get('event_frame_partition_complete')} / {c.get('event_time_coverage_complete')} |",
           f"| 已提取知识 | {c['knowledge_units']} 条；Codex 源证据复核完成：{c.get('knowledge_review_complete')} |",
           f"| 第二轮逐事件复核 | {c['events_second_pass_complete']} |",
           f"| 本地 VLM 无人复核使用 | {quality.get('local_vlm_unattended_use_approved',False)} |",
           "| 人工 CER/WER、时间边界误差、知识召回 | 尚未测量，不报告达标 |","",
           "校订转写共有 20 个句段、316 个对齐单元，原稿与原始对齐另存。"
           "原音轨完整解码得到 3,304,448 个 48kHz 样本，3,227 个音频帧，未发现时间间断。"
           "16kHz ASR 副本的长度差来自重采样的样本舍入。","",
           "## 怎样证明逐帧输入","",
           "- 原文件 hash 绑定到分 P；源帧保留实际 PTS 和时间基准。",
           "- FFmpeg 帧 hash 与 ffprobe 枚举逐条对应；Apple 原生 OCR、PyAV 再核对帧时间。",
           "- 使用已解码的完整 numpy 帧序列，直接调用本地视频处理器，绕过按 FPS 抽样的文件入口。",
           "- 原模型每两个连续帧构成一个时序 patch；两帧的像素均在模型输入中，不是只选其中一帧。",
           "- 实际模型消费的是已检查并保存 hash 的像素张量、媒体 token 与图像网格。",
           f"- 当前成功请求中源帧实际出现 {source_occurrences} 次（包含窗口重叠），额外补齐复制帧 {pad} 次；唯一源帧单独计数。",
           f"- 视觉输入的实际缩放尺寸集合为 {shapes}（宽、高）；OCR 和证据图保留原始 1080P。"
           "没有声称模型以原分辨率理解了每个像素。",
           "- 单帧闪现、奇数尾帧、不同间隔时间戳的测试验证了输入保留；这些不是模型识别准确率的证明。","",
           "## 已测量的处理时间","",
           "| 阶段 | 实测 | 说明 |","|---|---:|---|",
           f"| 媒体准备与校验 | {m['prepare_wall_seconds']:.2f} 秒 | 全帧枚举、hash、混流和音频副本等 |",
           f"| 全帧 OCR | {ocr.get('elapsed',0):.2f} 秒 | 原尺寸 2,065 帧 |",
           f"| 独立 ASR | {speech.get('asr_wall_seconds',0):.2f} 秒 | 包含该阶段模型加载 |",
           f"| 原稿强制对齐 | {speech.get('aligner_wall_seconds',0):.2f} 秒 | ASR 卸载后执行 |",
           f"| 校订稿重新对齐 | {corrected.get('wall_seconds',0):.2f} 秒 | 原稿不覆盖 |"]
    if vision.get("strict_input_complete"):
        lines += [f"| 最终配置视觉阶段 | {vision['total_wall_seconds_including_runs']:.2f} 秒 | 包括首批验证和续跑加载；开发期旧配置另计 |",
                  f"| 最终有效唯一帧吞吐 | {vision['effective_unique_frames_per_second']:.3f} 帧/秒 | 已含重叠、预处理、生成和落盘 |",
                  f"| MLX 已分配内存峰值 | {vision['peak_memory_gb']:.3f} GB | 不是全系统 RAM 占用 |",
                  f"| 进程峰值 RSS | {vision.get('process_peak_rss_gb',0):.3f} GB | macOS 进程统计口径 |"]
    else:lines += ["| 最终配置视觉阶段 | 进行中 | 全片最终指标尚未生成 |"]
    lines += ["",
              "ASR 对该样本的实时系数为 "
              f"{speech.get('asr_realtime_factor',0):.3f}。模型下载、代码开发、旧配置调试和 Codex 知识校订不计入上表的最终配置耗时。","",
              "## 选帧输入路线的对照","",
              "| 配置 | 初始唯一源帧 | 累计秒数 | 有效唯一帧/秒 | 备注 |","|---|---:|---:|---:|---|"]
    baselines=[("8 帧多图","window-8","初始窗口基线，工作量不同"),
               ("32 帧多图","window-32-images","其中一窗照抄示例，仅作性能诊断"),
               ("32 帧连续视频","temporal-v1-development","同一初始 92 帧；不做 FPS 抽样")]
    for label,folder,note in baselines:
        v=load(d/"benchmarks"/folder/"vision-metrics.json",{})
        # Development directory metrics is the initial three-window measurement.
        if v:
            lines.append(f"| {label} | {v['successful_unique_frames']} | {v['total_wall_seconds_including_runs']:.2f} | {v['effective_unique_frames_per_second']:.3f} | {note} |")
    lines += ["",
              "32 帧多图与连续视频的对照使用同一开头帧区间和像素预算。连续视频路线避免将每张独立图片再复制补齐时序维度。"
              "输出内容和后续结构化策略有差异，因此这不是所有素材上的通用加速保证。","",
              "## 修正与已知限制","",
              "1. ASR 的 Godot 误听和三项设置断句已依据原片文字校订并重对齐；原稿、原对齐和修订理由均保留。",
              "2. 开发期 JSON 截断、模板复述和类型错误已经记录；最终配置使用已安装的 llguidance 对字段、帧号枚举及数组长度作生成约束。",
              "3. 当前模型曾把 25 秒的游戏运行窗口误认成项目设置窗口，关键知识已按真实截图纠正。",
              "4. 观察摘要中的长设置名可能不完整；完整参数从原尺寸 OCR 和原图读取，短摘要不作为精确代码来源。",
              "5. 模型也可能把作者的操作要求写成已经发生的动作。K0006 保留作者陈述归属，不把静态菜单证明为已取消选项。",
              "6. 非语音音效没有独立听辨；未转写的间隔不自动等于静音。",
              "7. 原片没有展开碎片算法、动画轨道和代码；笔记保留这些缺口。",
              "8. 原浏览器任务空间后来不可用，21 秒播放器跳转的二次复核没有完成。已有媒体 hash、页面 BV/CID、全部 PTS 和内容抽检保留。",
              "9. Ego Lite 曾提示有更新，本轮未执行升级。",
              "10. 程序目前是单条试运行工具，输入限制为单分 P、音频不超过 180 秒；不是已经可运行 739 条收藏的完整调度器。","",
              "## 恢复与验证","",
              "- 主动中断一次后确认 8 个成功窗口回执逐字节不变，失败或未提交窗口重新处理。",
              "- 模型/媒体/窗口/像素/OCR/提示词身份不同的回执不混用；更换提示词的旧视觉试跑单独归档。",
              "- 原有 12 项相关检查通过；针对实际格式问题补充的 2 项检查及非法帧引用检查也已实际执行通过。"
              "仅复用未受影响的有效检查结果，没有因为等待推理而重跑全部测试。",
              "- 逐事件第二轮复核和源证据修改记录保存在 events.jsonl、knowledge.jsonl 与 quality-issues.json。","",
              "## 存放与下一步","",
              "环境、下载缓存、模型和媒体均位于外接固态盘当前项目中。模型及配置实际约 6.85GB，"
              "独立 Python 环境目录逻辑大小约 867MiB，原媒体约 6.23MB；最终用户包不包含模型权重。",
              ""]
    if vision.get("strict_input_complete"):
        fps=vision["effective_unique_frames_per_second"]
        days=20164710/fps/86400
        lines += [f"按前轮 672,157 秒时长、假定 30 FPS，并仅套用本条视觉吞吐，视觉阶段粗算约 {days:.1f} 天。"
                  "这是单一样本的数学外推，尚未展开所有实际分 P，不是全库工期承诺；OCR、知识复核和其他工作另计。",""]
    lines += ["当前应先解决自动语义误判和指令/动作归属，再用代表片段比较处理质量与算力。"
              "不建议直接启动整库本机严格逐帧任务；外部模型、GPU 费用、媒体传输或改用加速采样标准都需要另作明确决定。","",
              f"本条当前处理状态：{c['status']}。完整输入及知识产物是否就绪：{bool(finished)}。"
              "这不等于通过无人复核的模型质量验收。"]
    (root/"试运行报告.md").write_text("\n".join(lines).replace("True","是").replace("False","否")+"\n")
    print(json.dumps({"report":str(root/"试运行报告.md"),"pilot_artifacts_ready":bool(finished)},ensure_ascii=False))


if __name__=="__main__":main()
