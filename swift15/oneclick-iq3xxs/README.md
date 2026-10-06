# Swift 1.5 IQ3XXS 开发启动器

**简体中文 | [English](README.en.md)**

与 IQ2S 共用 [`../engine/`](../engine/) 源码。这里公开已在 RTX 3060 12GB 上检查的本地开发配置，尚未发布独立网盘懒人包。

默认 rk4v4、关闭 CUDA Graph、CPU 看图、自动 CPU 线程；33K 驻留窗口、128K 逻辑上下文、API 申请上界随驻留容量变化，MTP-Q4 / draft3 / 完整输出头。系统、developer 和工具定义按实际前缀保留；实际检索和输出看 `[kvmem-alloc]` 日志。API 上界不等于物理输出预算，未完成连续 32K／64K 输出测试。

将编译好的 Swift 引擎及所需 DLL 放入 `engine/`，MTP-Q4 模型 `swift15_iq3xxs_mtpq4.ninfer` 放入 `model/`。也可设置 `NINFER_IQ3_ENGINE`、`NINFER_IQ3_MODEL` 指向已有文件。默认核对已验收构建的引擎哈希；自行重新编译后，用 `NINFER_IQ3_ENGINE_SHA256` 指定自己构建的 SHA256。`capacity-policy.json` 是这次硬件测量快照，可用 `NINFER_IQ3_POLICY` 指定自己的容量记录；测量数值不保证其他后台负载或硬件也能容纳。

`启动.bat` 打开菜单，`预览参数.bat` 只显示参数。当前只允许 rk4/noGraph/300MiB 测量档启动，其余档位可预览。脚本不会自动下载模型或停止其他推理进程。

启动服务后，任意 OpenAI／Anthropic 兼容客户端均可连接 `http://127.0.0.1:8084`，模型 ID 为 `swift15-iq3xxs`，逻辑上下文 131072。API 上界随驻留容量变化，实际输出由每轮完整前缀和检索预算决定；包内不附测试专用 DSH 入口或预设。

CPU 视觉编码仍需语言模型、KV 和最终图像特征的显存。短工具链已验收；完整 Graph 矩阵、新窗口长压力和连续 32K／64K 输出没有全部完成。修复记录见 [`../../CHANGELOG.md`](../../CHANGELOG.md)。
