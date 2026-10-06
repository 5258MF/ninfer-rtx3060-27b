# Swift 1.5 IQ3XXS 开发启动器

**简体中文 | [English](README.en.md)**

与 IQ2S 共用 [`../engine/`](../engine/) 源码。这里公开已在 RTX 3060 12GB 上检查的本地开发配置，尚未发布独立网盘懒人包。

默认 rk4v4、关闭 CUDA Graph、CPU 看图、自动 CPU 线程；33K 驻留窗口、128K 逻辑上下文、API 默认输出上限 16K，MTP-Q4 / draft3 / 完整输出头。系统、developer 和工具定义按实际前缀保留；实际检索和输出看 `[kvmem-alloc]` 日志。16K 是 API 上限，未完成连续 16K 输出测试。

将编译好的 Swift 引擎及所需 DLL 放入 `engine/`，MTP-Q4 模型 `swift15_iq3xxs_mtpq4.ninfer` 放入 `model/`。也可设置 `NINFER_IQ3_ENGINE`、`NINFER_IQ3_MODEL` 指向已有文件。默认核对已验收构建的引擎哈希；自行重新编译后，用 `NINFER_IQ3_ENGINE_SHA256` 指定自己构建的 SHA256。`capacity-policy.json` 是这次硬件测量快照，可用 `NINFER_IQ3_POLICY` 指定自己的容量记录；测量数值不保证其他后台负载或硬件也能容纳。

`启动.bat` 打开菜单，`预览参数.bat` 只显示参数。当前只允许 rk4/noGraph/300MiB 测量档启动，其余档位可预览。脚本不会自动下载模型或停止其他推理进程。

启动服务后可用 `DSH-Minimal.cmd` 或 `DSH-Standard.cmd`。先安装 Python、Node 和 DSH，将 `DSH_MODULES_DIR` 设置为含 `dsh/`、`dsh-web-app/` 的 `@deepseek-ai` 目录；Node 从 PATH 查找，也可用 `DSH_NODE_EXE` 指定。生成的配置和会话在 `dsh-local/`，不修改全局 DSH 设置。普通预设使用 8K 压缩预留和 4K 摘要上限。

CPU 视觉编码仍需语言模型、KV 和最终图像特征的显存。短工具链已验收；完整 Graph 矩阵、新窗口长压力和连续 16K 输出没有全部完成。修复记录见 [`../../CHANGELOG.md`](../../CHANGELOG.md)。
