# Swift 1.5 源码与启动器

本目录提供 Swift 1.5 方案的完整引擎源码、Windows 构建脚本、启动器和模型转换工具。上游基线为 Ryan-gsq 分支 `b06908b`，KVMem、SM 数量识别及启动修复已合入源码。

## 目录

| 路径 | 内容 |
|---|---|
| [`engine/`](engine/) | 完整引擎：CMake、实现、应用、测试和第三方源码 |
| [`scripts/build-sm86.bat`](scripts/build-sm86.bat) | Windows / sm_86 构建脚本 |
| [`oneclick/`](oneclick/) | 懒人包启动器、测试脚本、设置和使用说明 |
| [`engine/tools/mtpq4.py`](engine/tools/mtpq4.py) | MTP 草稿头量化工具 |
| [`engine/tools/mkpatch.py`](engine/tools/mkpatch.py) | 生成模型转换用的 `ops.txt` / `lit.bin` |
| [`hotfix/swift15-hotfix-kv-capacity.zip`](hotfix/swift15-hotfix-kv-capacity.zip) | 旧包容量错误的启动器补丁，适用范围见故障排查 |

## 开始使用

- 下载懒人包：回到[项目首页](../README.md#下载)。
- 配置和运行：[使用指南](../docs/usage.md)。
- 从源码构建：[编译与开发](../docs/build.md)。
- 了解内部改动：[技术实现](../docs/implementation.md)。
- 查看报错处理：[故障排查](../docs/troubleshooting.md)。
- 查看修复版本：[更新记录](../CHANGELOG.md)。

源码许可见 [LICENSE](LICENSE)。模型来源和模型许可见[来源与许可](../docs/credits.md)。
