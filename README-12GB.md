# Bonsai2 12G 版

在 Windows 上使用 RTX 3060 12GB 运行 Swift-Bonsai-2 27B 三元模型。模型文件约 7.7 GiB，支持图片输入、工具调用和 KVMem 长对话。

## 下载与开始

下载 `ninfer-3060-12g-oneclick.zip`，网盘链接和提取码见[项目首页](README.md#下载)。现有网盘包为 2026-10-01 版；2026-10-03 的本机新构建待发布，差异见[更新记录](CHANGELOG.md)。

需要 RTX 30 系显卡，参数按 3060 12GB 调整。Windows 10 / 11，内存建议 32 GB，磁盘预留约 10 GB。模型不在包内，首次运行自动下载。

完整解压后双击 `启动.bat`，回车使用当前配置，按 C 打开向导；出现 `listening` 后运行 `测试.bat`，按 `接入信息.txt` 连接客户端。

## 模式

下面是当前已发布包的默认配置：

| 模式 | 总上下文 | 默认单次输出 |
|---|---:|---:|
| 普通 int8 | 88K，全部放显存 | 32K |
| rk8v4 | 112K，全部放显存 | 32K |
| KVMem | 256K，历史主要放内存 | 32K |
| KVMem + rk8v4 | 256K，历史主要放内存 | 32K |

KVMem 每轮读取选中的历史，配置上限不保证所有远处细节都能召回。显存上限和长文测试见[测试结果](docs/benchmarks.md#bonsai2-12g)。

## 详细资料

- [12G 使用指南](docs/bonsai2-12gb.md)：模式、参数和启动流程。
- [测试结果与限制](docs/benchmarks.md#bonsai2-12g)：速度、显存和长文结果。
- [技术实现](docs/implementation.md#bonsai2-12g)：移植与优化说明。
- [故障排查](docs/troubleshooting.md)、[更新记录](CHANGELOG.md)、[来源与许可](docs/credits.md)。

12GB 以上显卡也可以选择[Swift 1.5 方案](README.md)。8GB 显卡见[8G 版](README-8GB.md)。
