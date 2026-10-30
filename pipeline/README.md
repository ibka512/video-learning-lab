# 实验入口索引

接手先读 [交接说明](../docs/HANDOFF.md)，运行方式见 [环境与复现](../docs/REPRODUCIBILITY.md)。

| 模块 | 主要入口 |
|---|---|
| OCR局部路由 | `ocr_routing.py`、`prepare_ocr_review.py`、`run_ocr_review.py` |
| 原子事实与来源 | `fact_ledger.py`、`prepare_fact_packets.py`、`run_fact_pipeline.py` |
| 事实复核导出 | `finalize_fact_ledger.py`、`export_fact_markdown.py` |
| v2知识抽取 | `knowledge_schema.py`、`run_knowledge_v2.py`、`run_knowledge_pipeline.py` |
| v3步骤关系 | `learning_schema.py`、`run_learning_v3.py`、`finalize_learning_v3.py` |
| v4声画/反转对照 | `run_video_understanding_v4.py`、`evaluate_video_understanding_v4.py` |
| v5状态与补读 | `visual_ledger_v5.py`、`run_visual_ledger_v5.py` |
| v6属性与证据追加 | `object_evidence_v6.py`、`run_object_properties_v6.py` |
| v6评分与Markdown | `evaluate_object_properties_v6.py`、`export_object_properties_v6.py` |

历史实验入口保留其版本化数据协议，并非统一产品CLI。`finish_ocr_review.py`等历史流程入口有特定目录约定，使用前先检查本地输入，不要通过批量导入所有模块来探测功能。

当前视觉候选尚不能自动进入正式知识库。v5回看策略只保留作诊断；v6通过追加观察保留初值，但不自动裁决语义冲突。
