# Swift 1.5 IQ3XXS 懒人包

**简体中文 | [English](README.en.md)**

当前版本 **0.1.4**，与 IQ2S 共用 [Swift 引擎源码](../engine/)。独立懒人包已上传，下载入口见[项目首页](../../README.md#下载)。

完整解压后双击 `启动.bat`。首次运行下载 `swift15_iq3xxs_mtp.ninfer`，校验 SHA256 后用包内补丁转换为 `swift15_iq3xxs_mtpq4.ninfer`；无需 Python，需约 21 GiB 空闲磁盘。已有转换后模型可放进 `model/`，或设置 `NINFER_IQ3_MODEL` 指向它。模型与二进制依赖随网盘包准备，仓库公开启动器和完整引擎源码。

默认 RK4V4、关闭 CUDA Graph、CPU 视觉与自动 CPU 线程、128K 总上下文。驻留选择默认为 `auto`，按当前空闲显存估算上限；按 C 可改为推荐值或手动容量。系统／工具前缀、检索和输出由引擎逐请求分配，实际预算见 `[kvmem-alloc]` 日志。

`启动.bat dryrun` 或 `预览参数.bat` 只预览参数；服务启动后用 `测试.bat` 检查接口。默认接口为 `http://127.0.0.1:8084/v1`，模型 ID 为 `qwen3.8-27b`。包内不附测试专用 DSH 入口或会话状态。自编译引擎可用 `NINFER_IQ3_ENGINE` 与 `NINFER_IQ3_ENGINE_SHA256` 指定路径及哈希。

2026-10-13 已通过短对话、超大输出申请、多工具调用、CPU 看图及编辑历史请求，并验证首次下载的断点续传、SHA256 和真实模型的 PowerShell 转换。没有据此确认极限上下文或完整 32K／64K 长输出。CPU 视觉仍需最终图像特征、语言模型和 KV 使用 GPU。
