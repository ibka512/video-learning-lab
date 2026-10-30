# 早期本地试运行工具

保留用于媒体准备、转写/对齐、Apple Vision OCR、视觉输入打包、源审计和笔记导出的早期代码。

- `prepare.py`：原帧与时间身份、音频准备。
- `transcribe.py`：独立ASR与强制对齐。
- `ocr_frames.swift`：Apple Vision本地OCR。
- `vision.py`：实际媒体输入打包与试运行。
- `download_models.py`：按批准的清单下载固定版本模型；不会由基础测试自动运行。

代码需要对应的本地素材、环境与参数。公共依赖记录在 `environments/`，接手说明在 `docs/`。公共v4–v6推理入口引用本目录的视觉打包函数。
