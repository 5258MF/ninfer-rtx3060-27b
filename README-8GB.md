# Bonsai2 8G 版

面向 RTX 30 系 8GB 显卡的 Windows 推理方案，使用 Swift-Bonsai-2 27B ptq1 模型，模型下载约 6.6 GiB，MTP 草稿头转换为 Q4。

**测试使用 RTX 3060 12GB，并限制引擎显存占用；没有在真实 8GB 显卡上验证。** 实际 8GB 显卡的速度和桌面显存余量可能不同，详见[测试范围与限制](docs/benchmarks.md#bonsai2-8g)。

## 下载与开始

下载 `ninfer-3060-8g-oneclick.zip`，网盘链接和提取码见[项目首页](README.md#下载)。

需要 Windows 10 / 11、支持 CUDA 13 的 NVIDIA 驱动，内存建议 32 GB，第一次转换需约 15 GB 空闲磁盘。完整解压后双击 `启动.bat`，第一次会下载并转换模型；出现 `listening` 后运行 `测试.bat`。

## 模式

| 模式 | 总上下文 | 默认单次输出 |
|---|---:|---:|
| 普通，rk8v4 | 56K，全部放显存 | 32K |
| KVMem，rk8v4 | 256K，历史主要放内存 | 16K |
| 普通 rk4v4 | 84K，全部放显存 | 32K |
| KVMem + rk4v4 | 256K，历史主要放内存 | 32K |

默认使用 KVMem / rk8v4。rk4v4 虽然能容纳更多内容，但该版本的原有测试显示速度和质量有一定代价，见[使用指南](docs/bonsai2-8gb.md)。

## 详细资料

- [8G 使用指南](docs/bonsai2-8gb.md)：显存档位、模式与模型准备。
- [测试结果与限制](docs/benchmarks.md#bonsai2-8g)：显存峰值和测试方法。
- [技术实现](docs/implementation.md#bonsai2-8g)：ptq1、MTP Q4 和省显存方案。
- [故障排查](docs/troubleshooting.md)、[来源与许可](docs/credits.md)。

12GB 显卡见[12G 版](README-12GB.md)或[Swift 1.5 方案](README.md)。
