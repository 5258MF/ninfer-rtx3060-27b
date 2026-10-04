# Bonsai2 12G 源码与启动器

**简体中文 | [English](README.en.md)**

2026-10-04：main 已增加启动与验收脚本修复；0.1.0 Release 保留原始快照。缩窗后同步客户端限制，按 listening 判断启动完成，验收以 0/1/2 区分通过、失败和未确认。修复后的懒人包等待重新上传，旧网盘链接仍对应旧包。详情见[更新记录](../CHANGELOG.md)。

本目录公开 12GB 方案的完整引擎源码（原 ninfer-tree）、构建脚本和懒人包启动器，包含此前的 sm_86、KVMem、rk8v4、显存预算等改动，以及 2026-10-04 的三项智能体兼容修复。

当前源码版本为 **0.1.0**，见 [`VERSION`](VERSION)。Bonsai2 12G 已有[独立的 0.1.0 源码 Release](https://github.com/5258MF/ninfer-rtx3060-27b/releases/tag/bonsai2-12g-v0.1.0)，附件只包含本方案源码。百度、夸克懒人包见[项目首页](../README.md#下载)。

## 目录

| 路径 | 内容 |
|---|---|
| [`engine/`](engine/) | 完整 CMake 工程、实现、应用、测试、工具和第三方源码 |
| [`scripts/build-sm86.bat`](scripts/build-sm86.bat) | Windows / sm_86 构建脚本，源码和输出路径相对仓库定位 |
| [`oneclick/`](oneclick/) | 启动器、测试、公开默认配置、功耗设置和验收脚本 |
| [`engine/LICENSE`](engine/LICENSE) | 引擎代码许可；第三方目录保留各自许可 |
| [`engine/NOTICE`](engine/NOTICE) | 上游声明 |

## 使用与开发

- 下载和运行、参数、测试范围与从源码构建：[12G README](../README-12GB.md)。
- 新增修复与发布状态：[更新记录](../CHANGELOG.md)。
- Swift 1.5 使用独立的新引擎源码树，见 [swift15/](../swift15/README.md)。两种模型的引擎和参数不能混用。

源码不含模型、运行时 DLL 或编译产物。看图构建需要准备 FFmpeg 开发包，按 [12G README 的编译说明](../README-12GB.md#源码与编译)放置依赖。包内校验脚本对应指定发布构建，自行编译和重新打包后需同步更新哈希。
