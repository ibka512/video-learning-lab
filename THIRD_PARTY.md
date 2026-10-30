# 第三方来源

项目调用外部依赖和模型，未在仓库中重分发模型权重或依赖源码。

| 组件 | 用途 | 来源 |
|---|---|---|
| MLX-VLM | 本地视觉语言推理、结构化输出 | https://github.com/Blaizzy/mlx-vlm |
| MLX-Audio | ASR、强制对齐 | https://github.com/Blaizzy/mlx-audio |
| PyAV | 解码与原帧时间 | https://github.com/PyAV-Org/PyAV |
| PaddleOCR | 局部OCR对照 | https://github.com/PaddlePaddle/PaddleOCR |
| Pydantic | 结构校验 | https://github.com/pydantic/pydantic |
| Pillow / NumPy | 图像与张量准备 | https://github.com/python-pillow/Pillow · https://github.com/numpy/numpy |

模型仓库与固定版本列于 `environments/models.json`；该清单记录试验时的元数据，使用时仍应核对各来源的适用条款。相关项目借鉴记录见 [GitHub调研](docs/design/github-research.md)。

`support/pilot` 是本项目早期实验工具，不是上述项目的完整分发副本。原作者代码与模型的权利不会因本仓库公开而改变。项目本身的代码许可证尚待维护者指定。
