# 故障排查

[项目首页](../README.md) · [使用指南](usage.md) · [更新记录与版本识别](../CHANGELOG.md)

本页的两项启动修复针对 Swift 1.5 Q2S 包。Bonsai2 8G / 12G 的模式和容量设置见各自使用指南。

## KV 页数超出容量曲线

报错：`Main KV page count is outside the target capacity curve`。

旧引擎按请求的读入分段估算容量，而实际分段会按 SM 数对齐，例如 512 变成 640，导致页数低于规划器下限。[当前源码](../swift15/engine/src/runtime/engine/model_instance.cpp)已使用规划器容量曲线的最低页数。

更新引擎后继续使用启动器生成的参数。KVMem 的总上下文在内存，显存只保留窗口；`--max-context 204800` 大于 `--kvmem-window-pages 1152` 对应的窗口是正常配置，不需要因此删除窗口参数。

旧包也可以使用[容量启动器补丁](../swift15/hotfix/swift15-hotfix-kv-capacity.zip)：关闭引擎，解压到原包目录覆盖 `launcher/launch.ps1`。它只在遇到这条错误时关闭分段对齐并重试一次，不能修复下面的首次校准问题。

## 首次显卡校准失败

报错：`small_t_i8: this NINFER_SLIM_3060 build supports only int8 / rk8v4 / rk4v4 KV caches`。

精简构建没有编译 E8 内核，但旧自动校准仍会测试它们。显卡没有匹配校准记录时，即使用默认 rk4v4，也可能在加载模型前失败。[当前源码](../swift15/engine/src/calibration/device_calibration.cu)已让精简版只校准实际编译的 int8 / rk8v4 / rk4v4。

优先使用包含修复的新引擎，识别信息见更新记录。旧包可先在包目录的 CMD 窗口运行：

```bat
set "L8084_EXTRA_ARGS=--device-profile off"
启动.bat last
```

这会跳过路线校准，SM 数量仍会识别。更新引擎后，在新 CMD 窗口正常运行 `启动.bat` 即可。首次正常校准可能需要一段时间，请等待 `listening`。

## VC 运行库或其他 DLL 加载错误

若 Windows 提示 DLL 找不到或入口点不存在，先确认完整解压，保持 EXE 与所有 DLL 在 `engine` 目录中。所需 VC 文件包括 `msvcp140.dll`、`vcruntime140.dll`、`vcruntime140_1.dll`。

需要修复系统运行库时，使用[微软官方 x64 VC 运行库](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)。新版包已经检查静态及延迟导入依赖，并实际验证这三个 VC DLL 从包目录加载；[微软本地部署说明](https://learn.microsoft.com/en-us/cpp/windows/determining-which-dlls-to-redistribute)允许这种同目录部署方式。

DLL 加载错误与上面的 E8 校准逻辑错误分别处理。安装运行库不会补回精简构建没有编译的 E8 内核。

## 显存或主机内存分配失败

| 情况 | 处理 |
|---|---|
| 显存分配失败或窗口放不下 | 关闭其他占显存程序；按 R 重新读空闲显存，或按 C 调小单次输出 |
| 运行后突然很慢 | 检查是否显存被其他程序占用，释放后重新启动 |
| `cudaMallocHost` / `bad allocation` | 检查主机内存和其他模型占用，可调小总上下文；缩显存窗口不能覆盖所有主机锁页失败 |
| 已有模型运行或端口占用 | 关闭原模型，或修改 `设置.ini` 的 `PORT` |

启动器只处理它识别的容量报错，不能保证所有分配失败都自动恢复。输入很长的材料时，也需要考虑[长文召回范围](benchmarks.md#swift-15)。

## 下载、转换与客户端

- 下载中断：重新启动，已有 `.part` 文件会继续下载。
- 校验失败：检查模型文件是否完整，按包内说明重新下载。
- 补丁缺失：重新完整解压 `patch` 目录；现成的转换后模型可以继续使用。
- 客户端输出截断：最大输出按本次生成的 `接入信息.txt` 填写；思考也占输出预算。
- dsh 改完不生效：退出程序与托盘图标后再打开，避免旧配置写回。
- 窗口闪退：在包目录地址栏输入 `cmd`，再运行 `启动.bat` 查看具体错误。

引擎日志在包内 `logs` 目录。分享错误信息时，去掉 API Key 等私人信息。
