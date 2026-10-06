# Bonsai2 8G / 12G 共用源码与启动器

**简体中文 | [English](README.en.md)**

当前共用源码版本为 **0.1.5**，包含 CPU 检索索引、分块打分与按请求分配窗口。8G 和 12G 懒人包仍为 **0.1.3**，共用[项目首页](../README.md#下载)中的同一组百度、夸克链接。详情见[更新记录](../CHANGELOG.md)。

本目录公开 Bonsai2 8GB 与 12GB 共用的完整引擎源码（原 ninfer-tree）和构建脚本，并分别提供两种显存方案的懒人包启动器。两套方案使用不同的模型文件与启动配置；引擎实现和构建入口共用，通用修复也适用于两者。这里包含 sm_86、KVMem 与 rk8v4 相关改动，也涵盖显存预算和 2026-10-04 的三项智能体兼容修复。

当前 main 源码版本为 **0.1.5**，见 `VERSION`。本版本源码 Release 已撤回，main 保留待修正。历史 [Bonsai2 0.1.4 源码 Release](https://github.com/5258MF/ninfer-rtx3060-27b/releases/tag/bonsai2-12g-v0.1.4) 仍可下载；网盘懒人包版本不变。

## 目录

| 路径 | 内容 |
|---|---|
| [`engine/`](engine/) | 完整 CMake 工程与实现代码；应用与测试、工具及第三方源码也都包含在内 |
| [`scripts/build-sm86.bat`](scripts/build-sm86.bat) | Windows / sm_86 构建脚本，源码和输出路径相对仓库定位 |
| [`oneclick-8g/`](oneclick-8g/) | 8G 启动器、公开配置、验收与清单；和 12G 共用 `engine/` |
| [`oneclick-12g/`](oneclick-12g/) | 12G 启动器、公开默认配置和验收脚本；目录还包含测试与功耗设置 |
| [`engine/LICENSE`](engine/LICENSE) | 引擎代码许可；第三方目录保留各自许可 |
| [`engine/NOTICE`](engine/NOTICE) | 上游声明 |

## 使用与开发

- 下载和运行、参数、测试范围与从源码构建：[12G README](../README-12GB.md)。
- 新增修复与发布状态：[更新记录](../CHANGELOG.md)。
- Swift 1.5 使用独立的新引擎源码树，见 [swift15/](../swift15/README.md)。两种模型的引擎和参数不能混用。

源码不含模型、运行时 DLL 或编译产物。看图构建需要准备 FFmpeg 开发包，按 [12G README 的编译说明](../README-12GB.md#源码与编译)放置依赖。包内校验脚本对应指定发布构建，自行编译和重新打包后需同步更新哈希。

8G 与 12G 使用同一份引擎源码。默认构建生成 int8/rk8v4 引擎；8G 的 rk4v4 构建前设置 `NINFER_RK4_SM86=ON`，输出到独立的 `build-rk4/`。8G 使用方法见 [8G README](../README-8GB.md)。运行时 DLL 与 Q4 MTP 的 `patch/lit.bin` 仍需从懒人包准备，不进入源码仓库。
