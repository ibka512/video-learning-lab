# 环境与复现

## 1. 无素材基础回归

使用 Python 3.12。测试依赖只有 NumPy、Pillow、Pydantic，版本来自已验证环境。

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r environments/requirements-tests.txt
.venv/bin/python -m unittest discover -s pipeline -p 'test_*.py' -v
.venv/bin/python scripts/check_publication.py
```

默认跳过1项需要 Apple Silicon、MLX和已下载模型的真实输入检查。其余测试包含公开的 v5 覆盖失败最小样例。基础回归不联网获取媒体，不加载模型，不需要API密钥。

## 2. Apple Silicon本地输入检查

先准备自己授权的环境和模型目录。`mlx-requirements.in`/`.lock` 是历史记录，不会通过基础测试自动安装。安装或下载前检查资源、版本与相应授权。

```sh
export VIDEO_LEARNING_MODEL_DIR=/path/to/local/Qwen3-VL-4B-Instruct-4bit
python -m unittest discover -s pipeline -p 'test_object_evidence_v6.py' -v
```

`python` 在此应是包含MLX依赖的本地环境解释器。模型目录须已存在，测试使用本地文件，不负责下载。

## 3. 语义实验所需输入

公开仓库没有视频、截图、完整转写、全量标注或模型。完整历史成绩需要维护者保管的本地材料。不要把公开的汇总指标解释成完整可独立重算的数据集。

v6输入根目录至少包括：

```text
local-data/v6/
  protocol.json
  protocol-r2.json
  sources/tasks.public.jsonl
  sources/source-identity.json
  references/properties.json
  references/freeze.json
```

每条任务需有 `task_id`、`case_id`、`position`、`object_scope`、`properties` 和 `source`。`source` 包含源帧身份、PTS、time_base、时间、原图/裁切路径、边界和文件校验值。素材存放在本地未跟踪位置。

为自己的素材建立清单，校验路径、图片与时间关系；先固定评分目标，再推理。不要复用经过路径修改的旧请求身份或伪造下载回执。

推理入口读取模型目录中的版本回执 `pilot-download-receipt.json`。已有下载工具可记录批准过的模型版本；该回执和权重都不在公开仓库内。

```sh
python pipeline/run_object_properties_v6.py \
  --root local-data/v6 \
  --out local-data/v6/new-run \
  --model "$VIDEO_LEARNING_MODEL_DIR"

python pipeline/evaluate_object_properties_v6.py \
  --root local-data/v6 --out local-data/v6/new-run

python pipeline/export_object_properties_v6.py \
  --root local-data/v6 --out local-data/v6/new-run
```

评分参考仅由评分入口读取，不能用于补读答案选择。准备脚本位于 `experiments/`，仍需要相应版本的本地原始材料，不能单独从公共仓库构造全部参考。

## 4. 公开版与原始本地版的差异

- 公共推理入口使用 `support/pilot/vision.py`，不再依赖本地报告目录中的工具位置。
- 回归测试使用最小化失败样例，真实模型测试通过环境变量显式开启。
- 历史报告中本地证据链接变成说明文字；不上传原始素材与完整回执。
- 依赖锁移除机器专属编译命令注释，保留包版本与哈希。
- `provenance/source-map.json` 保存来源与公开文件的校验映射。公开版是适配后的快照，不声称其文件哈希与旧推理回执完全相同。

## 5. 记录约定

每次新运行记录代码、模型、输入、提示、参数、实际消费图像及原始输出。输入或代码改变即使用新目录，原始回执不覆盖。只在相关条件变化或出现具体缺口时重跑测试。
