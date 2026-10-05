# Swift 1.5 源码与启动器

**简体中文 | [English](README.en.md)**

当前源码版本为 **0.1.2**，已按模型分别发布源码 Release，包含按实际系统提示分配、8K 输出保底与共享缓存越界修复。网盘仍是此前上传的 **0.1.0** 包；新修复包等待重新上传，下载表中的旧链接未改成 0.1.2。详情见[更新记录](../CHANGELOG.md)。

本目录提供 Swift 1.5 方案的完整引擎源码和 Windows 构建脚本，启动器与模型转换工具也在本目录。上游基线为 Ryan-gsq 分支 `b06908b`，KVMem、SM 数量识别及启动修复已合入源码。

当前源码版本为 **0.1.2**，见 [`VERSION`](VERSION)。Swift 1.5 Q2S 已有[独立的 0.1.2 源码 Release](https://github.com/5258MF/ninfer-rtx3060-27b/releases/tag/swift15-v0.1.2)，附件只包含本方案源码。百度、夸克懒人包见[项目首页](../README.md#下载)。

2026-10-04 已同步最新 Q2S 源码与启动器，包含显存预算和内容打分；连接保活与推理控制也已同步，验收工具及多轮看图 KV 借页时序修复一并更新。

同日继续同步三项智能体兼容修改，分别保护中途分叉和压缩时的缓存复用，封顶输出与思考预算，并适配工具参数。当前实现的接口边界见[更新记录](../CHANGELOG.md#2026-10-04三项智能体兼容修改与-bonsai2-12g-源码公开)。Bonsai2 8G / 12G 共用源码另见 [bonsai2/](../bonsai2/README.md)。

## 目录

| 路径 | 内容 |
|---|---|
| [`engine/`](engine) | 完整 CMake 工程与实现代码；应用与测试、第三方源码也在其中 |
| [`scripts/build-sm86.bat`](scripts/build-sm86.bat) | Windows / sm_86 构建脚本 |
| [`oneclick/`](oneclick) | 懒人包启动器、测试脚本、设置和使用说明 |
| [`oneclick/verify-kit-manifest.ps1`](oneclick/verify-kit-manifest.ps1) | 指定发布构建的文件完整性与中文文件名检查 |
| [`oneclick/verify-arch-engine.ps1`](oneclick/verify-arch-engine.ps1) | 短提示、超窗正负对照、接口、工具结果回传、输出封顶与图片回归 |
| [`engine/tools/mtpq4.py`](engine/tools/mtpq4.py) | MTP 草稿头量化工具 |
| [`engine/tools/mkpatch.py`](engine/tools/mkpatch.py) | 生成模型转换用的 `ops.txt` / `lit.bin` |
| [`hotfix/swift15-hotfix-kv-capacity.zip`](hotfix/swift15-hotfix-kv-capacity.zip) | 旧包容量错误的启动器补丁，适用范围见更新记录 |

## 开始使用

- 下载懒人包：回到[项目首页](../README.md#下载)。
- 配置和运行：[使用指南](../README.md#swift-15-使用说明)。
- 从源码构建：[编译与开发](../README.md#源码与编译)。
- 了解内部改动：[技术实现](../README.md#技术实现)。
- 查看报错处理：[故障排查](../CHANGELOG.md#启动错误与处理)。
- 查看修复版本：[更新记录](../CHANGELOG.md)。

源码许可见 [LICENSE](LICENSE)。模型来源和模型许可见[来源与许可](../README.md#来源与许可)。
