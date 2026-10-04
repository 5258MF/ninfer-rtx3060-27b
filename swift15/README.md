# Swift 1.5 源码与启动器

**简体中文 | [English](README.en.md)**

2026-10-04：main 已增加启动与验收脚本修复；0.1.0 Release 保留原始快照。缩窗后同步客户端限制，按 listening 判断启动完成，验收以 0/1/2 区分通过、失败和未确认。修复后的懒人包等待重新上传，旧网盘链接仍对应旧包。详情见[更新记录](../CHANGELOG.md)。

本目录提供 Swift 1.5 方案的完整引擎源码、Windows 构建脚本、启动器和模型转换工具。上游基线为 Ryan-gsq 分支 `b06908b`，KVMem、SM 数量识别及启动修复已合入源码。

当前源码版本为 **0.1.0**，见 [`VERSION`](VERSION)。Swift 1.5 Q2S 已有[独立的 0.1.0 源码 Release](https://github.com/5258MF/ninfer-rtx3060-27b/releases/tag/swift15-v0.1.0)，附件只包含本方案源码。百度、夸克懒人包见[项目首页](../README.md#下载)。

2026-10-04 已同步最新 Q2S 源码与启动器，包括显存预算、内容打分、连接保活、推理控制、验收工具和多轮看图 KV 借页时序修复。

同日继续同步三项智能体兼容修改：中途分叉/压缩缓存复用保护、输出与思考预算封顶、工具参数适配。当前实现的接口边界见[更新记录](../CHANGELOG.md#2026-10-04三项智能体兼容修改与-bonsai2-12g-源码公开)。Bonsai2 12G 的完整源码另见 [bonsai2-12g/](../bonsai2-12g/README.md)。

## 目录

| 路径 | 内容 |
|---|---|
| [`engine/`](engine) | 完整引擎：CMake、实现、应用、测试和第三方源码 |
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
