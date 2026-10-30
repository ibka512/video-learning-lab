# 开发交接

更新：2026-10-30，当前版本 v6。

## 目标

视频 → 独立语音转写与对齐 → 画面/OCR观察 → 对象、状态、过程与讲解关系 → 有来源的知识 → 按视频及领域输出 AI 可读 Markdown。最终再扩展到收藏夹批处理和跨视频检索。

## 已完成

- 本地 Qwen3-ASR、ForcedAligner、Qwen3-VL 的试运行与版本记录。
- 原帧/PTS索引、实际输入校验、Apple Vision OCR与局部PaddleOCR实验。
- 原文引用及来源校验、源复核决定绑定、步骤遗漏与条件检查。
- 多图、单图、声画转写联合输入、反转呈现的对照实验。
- v6对象属性分离与证据追加机制；窗口身份与内容分开，属性可并存。
- 模型原始输出、程序处理、Codex复核修正分别保留的实验纪律。

## 当前结果与决定

v6两种输入均32/40：内容/勾选30/32，窗口身份2/8。28对图像张量相同，仅2项答案不同。内容识别能力与窗口身份判断能力差异明显。

v5回看后正确数从17/24降为16/24，错误由5增至8。这个版本未启用为默认修复。v6只追加证据，真实回归和本轮补读均验证原始记录不会被覆盖。

**没有实现自动可信知识入库。** 正式稿仍需要绑定证据的复核决定；新视觉稿是候选。不要把已能生成Markdown误认为已完整理解视频。

## 接手后的第一件事

执行基础测试和公开检查，再阅读 [下一轮任务](ROADMAP.md)。下一轮集中验证窗口标题、独立标题栏、边界与前景关系，分别控制目标区域和分辨率。

尚未授权下载新模型或新增外部服务。可先用现有模型准备受控实验，完成可审阅方案后再处理新增资源决定。

## 关键入口

| 工作 | 入口 |
|---|---|
| 语音与原帧准备 | `support/pilot/prepare.py`、`transcribe.py` |
| 视觉输入打包与历史逐帧实验 | `support/pilot/vision.py` |
| OCR局部路由 | `pipeline/ocr_routing.py`、`run_ocr_review.py` |
| 原子事实与源复核 | `fact_ledger.py`、`finalize_fact_ledger.py` |
| 通用知识抽取 | `run_knowledge_v2.py`、`run_knowledge_pipeline.py` |
| 步骤与条件 | `learning_schema.py`、`run_learning_v3.py` |
| v4对照 | `run_video_understanding_v4.py` |
| v5状态与回看诊断 | `visual_ledger_v5.py`、`run_visual_ledger_v5.py` |
| v6对象属性与追加记录 | `object_evidence_v6.py`、`run_object_properties_v6.py` |
| v6评分及导出 | `evaluate_object_properties_v6.py`、`export_object_properties_v6.py` |

以上 pipeline 文件均位于仓库 `pipeline/`。

## 本地证据交接

完整运行资料仍由维护者在外接盘保管。公开版移除了媒体、完整转写、登录/获取数据、绝对路径和大量原始回执。需要重跑语义实验时，由维护者提供相应的本地输入清单和授权素材；按 [复现说明](REPRODUCIBILITY.md) 生成新的运行目录。

两个视频均为开发回归：`BV1cyHR67EbJ`（Godot）和 `BV1Xato6REfH`（前端组件）。不要将第二个视频继续称为未见过的新样本。

## 提交与交接方式

每次提交围绕可判断的行为，记录问题、改动、验证及边界。更新 `DEVLOG.md` 和本页的当前状态。公开报告不得出现本机路径或原始素材，指标变化必须附明确口径。Git历史从首次公开导入开始，不补造早期提交时间。
