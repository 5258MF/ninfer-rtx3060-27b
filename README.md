# ninfer：在 RTX 3060 上本地运行 27B 模型

为 Windows 提供 Swift 1.5 和 Bonsai2 的本地推理方案，支持对话、图片输入、工具调用和 KVMem 长上下文。懒人包解压后双击启动，模型在第一次运行时下载。

## 选择版本

| 版本 | 显卡与显存 | 模型 | 入口 |
|---|---|---|---|
| **Swift 1.5（默认推荐）** | RTX 30 系，12 GB 以上；RTX 40 系可运行但未实测 | Swift-1.5 27B IQ2_S，约 9.55 GiB | [使用指南](docs/usage.md) |
| **Bonsai2 12G** | 按 RTX 3060 12GB 调整，面向 RTX 30 系 | Swift-Bonsai-2 三元模型，约 7.7 GiB | [12G 版说明](README-12GB.md) |
| **Bonsai2 8G** | 面向 RTX 30 系 8GB；测试在 12GB 卡限制显存完成 | Swift-Bonsai-2 ptq1，约 6.6 GiB | [8G 版说明](README-8GB.md) |

Swift 1.5 默认总上下文为 200K，最多可配置 256K；KVMem 会从历史中选取部分内容放入显存。容量上限与长文召回效果请一起看[测试结果与已知限制](docs/benchmarks.md)。

## 下载

懒人包不包含模型。请完整解压，保留 `engine`、`launcher` 和 `patch` 等目录。

| 版本 | 文件名 | 网盘 | 提取码 |
|---|---|---|---|
| Swift 1.5 | `ninfer-3060-swift15-q2s-mtp-oneclick.zip` | [百度网盘](https://pan.baidu.com/s/19jdHRmGizwVdNcXz8FAiRw) | `6m91` |
| Swift 1.5 | 同上 | [夸克网盘](https://pan.quark.cn/s/2bacd7e8c782) | `qiA3` |
| Bonsai2 12G | `ninfer-3060-12g-oneclick.zip` | [百度网盘](https://pan.baidu.com/s/1p4OL2EzR0h4iCD2Mu40Ecw) | `9vc5` |
| Bonsai2 8G | `ninfer-3060-8g-oneclick.zip` | [百度网盘](https://pan.baidu.com/s/1_ZIDNnMaaOGRk-YnHb0jUA) | `gkyy` |

**发布状态：Swift 1.5 的最新修复包已重新打包，网盘文件仍待更新；Bonsai2 12G 的新构建也尚待发布。** 当前链接和最新构建的区别、版本识别信息统一记录在 [CHANGELOG.md](CHANGELOG.md)。

## 快速开始

下面以 Swift 1.5 为例：

1. 准备 Windows 10 / 11、支持 CUDA 13 的 NVIDIA 驱动、12 GB 以上显存。内存建议 32 GB，第一次下载和转换模型需约 22 GB 空闲磁盘。
2. 下载懒人包，解压到简单路径，例如 `D:\ninfer-swift15`。
3. 双击 `启动.bat`，确认配置后按回车。第一次会下载、校验并转换模型；出现 `listening` 后即可使用。
4. 双击 `测试.bat` 检查回答。连接客户端时，按启动后生成的 `接入信息.txt` 填写接口和输出上限。

默认接口为 `http://127.0.0.1:8084/v1`，模型 ID 为 `qwen3.8-27b`。修改参数、模型准备和客户端接入见[使用指南](docs/usage.md)；启动失败按[故障排查](docs/troubleshooting.md)处理。

## 文档

| 想了解什么 | 文档 |
|---|---|
| Swift 1.5 的配置、模型准备和客户端接入 | [使用指南](docs/usage.md) |
| Bonsai2 两个版本怎么用 | [12G 使用指南](docs/bonsai2-12gb.md)、[8G 使用指南](docs/bonsai2-8gb.md) |
| 启动报错、运行库、显存和内存问题 | [故障排查](docs/troubleshooting.md) |
| 更新了什么、哪个包包含修复 | [更新记录](CHANGELOG.md) |
| 速度、困惑度、长文测试和适用范围 | [测试结果与已知限制](docs/benchmarks.md) |
| 显存优化、KVMem、检索和内核改动 | [技术实现](docs/implementation.md) |
| 如何编译和生成模型补丁 | [编译与开发](docs/build.md) |
| 模型来源、代码来源和许可证 | [来源与许可](docs/credits.md) |

## 源码

Swift 1.5 的完整引擎源码在 [`swift15/engine/`](swift15/engine/)，包含 CMake 工程、实现、应用、测试和第三方源码。启动器和设置在 [`swift15/oneclick/`](swift15/oneclick/)，构建脚本在 [`swift15/scripts/`](swift15/scripts/)。

入口见 [swift15/README.md](swift15/README.md)。Bonsai2 8G / 12G 的改造源码尚未整理进本仓库；下载方案和测试记录已公开。引擎代码与模型的许可分别见[来源与许可](docs/credits.md)。
