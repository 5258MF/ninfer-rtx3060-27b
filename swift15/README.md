# Swift 1.5 IQ2S / IQ3XXS 共用引擎源码

**简体中文 | [English](README.en.md)**

当前引擎源码版本为 **0.1.3**，由 IQ2_S 和 IQ3_XXS 共用；当前 Swift 懒人包仍为 **IQ2S 0.1.2**。IQ3XXS 懒人包尚未发布，后续会单独更新下载入口。Swift 懒人包与 Bonsai2 共用[项目首页](../README.md#下载)中的百度、夸克链接。详情见[更新记录](../CHANGELOG.md)。

本目录只有一套 Swift 1.5 引擎源码，覆盖 IQ2_S 与 IQ3_XXS 两种量化格式。量化类型、NInfer 格式映射和对应 GPU 内核都已纳入同一个 CMake 工程，不需要复制出第二套引擎目录。当前 oneclick-iq2s/ 内的启动器和设置仍针对 IQ2S 懒人包。

源码版本 **0.1.3** 见 VERSION，对应 [Swift 1.5 IQ2S / IQ3XXS 共用源码 Release](https://github.com/5258MF/ninfer-rtx3060-27b/releases/tag/swift15-v0.1.3)，附件只包含 swift15/ 目录。此前的 [0.1.2 Release](https://github.com/5258MF/ninfer-rtx3060-27b/releases/tag/swift15-v0.1.2) 保留为历史版本。现有 IQ2S 懒人包下载见[项目首页](../README.md#下载)。

2026-10-05 同步了 CPU 视觉编码实现：可选 OpenBLAS 加速、自动线程校准和缓存；OpenBLAS 不可用时回退到参考 CPU 实现。该引擎路径供 IQ2S 与 IQ3XXS 共用。交接记录中的 IQ2S、IQ3XXS 看图测试和 CPU 后端单测均已通过；这不代表所有图片、CPU 或长上下文配置都经过全面验证。

2026-10-04 同步了 Swift 引擎的显存预算、内容打分、连接保活和推理控制，并更新多轮看图 KV 借页时序处理。随后同步三项智能体兼容修改：保护中途分叉和压缩时的缓存复用、封顶输出与思考预算、适配工具参数。实现边界见[更新记录](../CHANGELOG.md#2026-10-04三项智能体兼容修改与-bonsai2-12g-源码公开)。Bonsai2 8G / 12G 共用源码见 [bonsai2/](../bonsai2/README.md)。

## 目录

| 路径 | 内容 |
|---|---|
| [engine/](engine) | 完整 CMake 工程与实现代码；应用、测试和第三方源码也在其中 |
| [scripts/build-sm86.bat](scripts/build-sm86.bat) | Windows / sm_86 构建脚本 |
| [oneclick-iq2s/](oneclick-iq2s) | 当前 IQ2S 懒人包启动器、测试脚本、设置和使用说明 |
| [oneclick-iq2s/verify-kit-manifest.ps1](oneclick-iq2s/verify-kit-manifest.ps1) | 指定发布构建的文件完整性与中文文件名检查 |
| [oneclick-iq2s/verify-arch-engine.ps1](oneclick-iq2s/verify-arch-engine.ps1) | 引擎接口、输出封顶、工具回传和看图回归 |
| [engine/tools/mtpq4.py](engine/tools/mtpq4.py) | MTP 草稿头量化工具 |
| [engine/tools/mkpatch.py](engine/tools/mkpatch.py) | 生成模型转换用的 ops.txt / lit.bin |
| [hotfix/swift15-hotfix-kv-capacity.zip](hotfix/swift15-hotfix-kv-capacity.zip) | 旧包容量错误的启动器补丁，适用范围见更新记录 |

## 开始使用

- 当前可下载的懒人包是 IQ2S 0.1.2：回到[项目首页](../README.md#下载)。
- 配置和运行：[使用指南](../README.md#swift-15-使用说明)。
- 从源码构建：[编译与开发](../README.md#源码与编译)。
- 了解内部改动：[技术实现](../README.md#技术实现)。
- 查看报错处理：[故障排查](../CHANGELOG.md#启动错误与处理)。
- 查看修复版本：[更新记录](../CHANGELOG.md)。

源码许可见 [LICENSE](LICENSE)。模型来源和模型许可见[来源与许可](../README.md#来源与许可)。
