# 来源与许可

[项目首页](../README.md) · [编译与开发](build.md)

代码许可与模型许可分别适用。Swift 1.5 引擎代码的 Apache-2.0 文本见 [swift15/LICENSE](../swift15/LICENSE)，源文件保留上游许可证和贡献者信息。

## Swift 1.5

- **Swift-1.5 Qwen3.8-27B GSQ-RCO**：UkisAI（Swift Open License v1.0，允许署名再分发；年收入超过 100 万美元的商用需要另外和 UkisAI 谈）；基座 Qwen3.8-27B（Apache-2.0）。
- **.ninfer 格式模型**：魔搭社区 fyb423 转换发布（[Swift-1.5-Qwen3.8-27B-GSQ-RCO-NInfer](https://www.modelscope.cn/models/fyb423/Swift-1.5-Qwen3.8-27B-GSQ-RCO-NInfer)）。
  - 本项目对模型的改动（Swift Open License 4(b) 要求的变更说明）：只把 MTP 草稿头的 7 个张量从 Q6_K 重新量化成 Q4（query/gate、output、MLP）和 Q8（key/value），其余权重逐字节不变。懒人包里只放补丁，不放模型本身，模型由用户从魔搭下载。
- **引擎**：Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco（Apache-2.0），它是 iamwavecut/ninfer-all 的分支，最早来自 Neroued/ninfer。
- **KVMem**：qzshch/ninfer-kvmem（Apache-2.0）的新引擎实现；kvmem-llama.cpp 项目和它的论文。
- 3090 分支和其他 3060 用户公开的调优记录，也提供了很多参考。

## Bonsai2 12G

- **Bonsai-2 27B 三元模型**：prism-ml；**Swift-Bonsai-2**：魔搭社区 fyb423 发布的版本；原版模型文件用的是魔搭社区 Lxt1992 的镜像。
- **ninfer 引擎**：原作者的三元推理引擎，以及作者提供的 3060 升级包和调优资料。
- **KVMem**：kvmem-llama.cpp 项目和它的论文。
- **rk8v4**：ninfer-all 项目的实现。
- 3090 分支和其他 3060 用户公开的调优记录，也提供了很多参考。

## Bonsai2 8G

- **Bonsai-2 27B 三元模型**：prism-ml；**Swift-Bonsai-2**（含 ptq1 版）：魔搭社区 fyb423 发布的版本。
- **ninfer 引擎**：原作者的三元推理引擎，以及作者提供的 3060 升级包和调优资料。
- **KVMem**：kvmem-llama.cpp 项目和它的论文。
- **rk8v4 / rk4v4**：ninfer-all 项目的实现。
- 3090 分支和其他 3060 用户公开的调优记录，也提供了很多参考。
