# Video Learning Lab

从视频中提取带时间和来源依据的知识，生成供 AI 使用的 Markdown。这里记录代码、开发过程、失败实验和交接信息。

**当前阶段：实验原型。** 内容属性已有部分可用能力，窗口身份和过程理解仍不可靠，尚不支持自动把整个收藏夹转成可信知识库。

## 从这里开始

- 接手开发：[交接说明](docs/HANDOFF.md)
- 了解进展：[开发日志](docs/DEVLOG.md) · [当前问题](docs/KNOWN_ISSUES.md)
- 运行代码：[环境与复现](docs/REPRODUCIBILITY.md)
- 理解结构：[架构与数据流](docs/ARCHITECTURE.md)
- 继续实验：[下一轮任务](docs/ROADMAP.md)
- 核对公开范围：[公开内容说明](docs/PUBLICATION.md)

## 最近一次结果：v6，2026-10-30

同一开发视频的24个选定画面位置，拆成28个任务、40项属性：

| 检查内容 | 原生单图 | 时序单帧 |
|---|---:|---:|
| 全部属性正确 | 32/40 | 32/40 |
| 内容及勾选属性正确 | 30/32 | 30/32 |
| 窗口身份正确 | 2/8 | 2/8 |
| 两个共存属性都识别出来 | 2/4 | 3/4 |

两种输入的28对图像张量相同，未显示整体准确率差异。回看采用只追加证据的方式，保留全部40个初始属性，分歧不会通过投票覆盖原记录。

这些是现有开发集和 Codex 源图复核参考下的结果，不是独立人工标注的新视频准确率。详见 [v6报告](docs/experiments/v6.md) 和 [机器可读结果](results/v6/evaluation.json)。

## 基础测试

Python 3.12，从仓库根目录运行：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r environments/requirements-tests.txt
.venv/bin/python -m unittest discover -s pipeline -p 'test_*.py' -v
```

基础测试使用公开的最小回归样例，不需要 B 站登录、视频素材、模型权重或 API 密钥。Apple Silicon 上的真实模型输入检查是可选项，见复现说明。

## 目录

```text
pipeline/           事实校验、知识抽取、v3–v6实验、测试
support/pilot/      早期语音、逐帧、Apple Vision OCR工具
experiments/        选帧、参考冻结等准备入口，需要本地输入
tests/fixtures/     无素材的最小失败回归样例
environments/       记录的依赖锁、模型版本清单
docs/               交接、开发日志、方案、公开版实验报告
results/            v4–v6汇总指标与验收决定
provenance/         本地源文件与公开文件的校验映射
scripts/            公开内容检查
```

本仓库不包含视频、截图、完整转写、收藏夹清单、账号数据、模型权重或本机缓存。完整证据由本地交接包保管；当前公开版本不能脱离这些输入重新验证全部历史语义成绩。

首次 Git 导入为当前状态快照。2026-10-26 至 2026-10-30 的开发过程通过日志和报告记录，没有伪造过去的逐日提交。维护者尚未指定项目代码许可证；第三方依赖和模型的来源列于 [THIRD_PARTY.md](THIRD_PARTY.md)。
