# ninfer：在 RTX 3060 上本地运行 27B 模型

**简体中文 | [English](README.en.md)**

当前源码：**Swift 1.5 Q2S 0.1.2，Bonsai2 8G / 12G 0.1.3**。Bonsai2 新增 CPU 检索索引与分块打分；三套方案统一按实际系统提示、检索目标和输出余量分配窗口。网盘链接仍指向原有包，新包待重新上传。详情见[更新记录](CHANGELOG.md)。

为 Windows 提供 Swift 1.5 和 Bonsai2 的本地推理方案，支持对话、图片输入、工具调用和 KVMem 长上下文。懒人包解压后双击启动，模型在第一次运行时下载。

## 选择版本

| 版本 | 显卡与显存 | 模型 | 入口 |
|---|---|---|---|
| **Swift 1.5（默认推荐）** | RTX 30 系，12 GB 以上；RTX 40 系可运行但未实测 | Swift-1.5 27B IQ2_S，约 9.55 GiB | [使用指南](#swift-15-使用说明) |
| **Bonsai2 12G** | 按 RTX 3060 12GB 调整，面向 RTX 30 系 | Swift-Bonsai-2 三元模型，约 7.7 GiB | [12G 版说明](README-12GB.md) |
| **Bonsai2 8G** | 面向 RTX 30 系 8GB；测试在 12GB 卡限制显存完成 | Swift-Bonsai-2 ptq1，约 6.6 GiB | [8G 版说明](README-8GB.md) |

Swift 1.5 默认总上下文为 200K，最多可配置 256K；KVMem 会从历史中选取部分内容放入显存。容量上限与长文召回效果请一起看[测试结果与已知限制](#测试结果与已知限制)。

## 下载

懒人包不包含模型。请完整解压，保留 `engine`、`launcher` 和 `patch` 等目录。

| 版本 | 文件名 | 网盘 | 提取码 |
|---|---|---|---|
| Swift 1.5 Q2S · 0.1.0 | `ninfer-3060-swift15-q2s-mtp-oneclick-0.1.0.zip` | [百度网盘](https://pan.baidu.com/s/1-t29Y9MZIHWliToRQI6i7g) | `t7f6` |
| Swift 1.5 Q2S · 0.1.0 | `ninfer-3060-swift15-q2s-mtp-oneclick-0.1.0.zip` | [夸克网盘](https://pan.quark.cn/s/8f383d9bb626) | `pnps` |
| Bonsai2 12G · 0.1.0 | `ninfer-3060-12g-bonsai2-oneclick-0.1.0.zip` | [百度网盘](https://pan.baidu.com/s/1-S14MappSh7uttKqIw1M_A) | `c62v` |
| Bonsai2 12G · 0.1.0 | 同上 | [夸克网盘](https://pan.quark.cn/s/b526dae6d411) | `aHfN` |
| Bonsai2 8G · 旧版 | `ninfer-3060-8g-oneclick.zip` | [百度网盘](https://pan.baidu.com/s/1_ZIDNnMaaOGRk-YnHb0jUA) | `gkyy` |

**发布状态：Swift 1.5 Q2S 与 Bonsai2 12G 的 0.1.0 包均已上传百度和夸克，包含 2026-10-04 同步的三项智能体兼容修改。** Swift 夸克链接已按用户最新提供的信息更正，请使用本表中的版本化文件名、链接和提取码。8G 版沿用之前的发布包。版本识别信息、修复记录和相关报错处理见 [CHANGELOG.md](CHANGELOG.md)。

源码按模型分别发布，每个 Release 都提供只包含对应源码目录的附件：[Swift 1.5 Q2S 源码 Release](https://github.com/5258MF/ninfer-rtx3060-27b/releases/tag/swift15-v0.1.2) 和 [Bonsai2 12G 源码 Release](https://github.com/5258MF/ninfer-rtx3060-27b/releases/tag/bonsai2-12g-v0.1.3)。Swift 的 `VERSION` 为 `0.1.2`，Bonsai2 为 `0.1.3`；后续开发继续在 `main` 更新，发布内容见 CHANGELOG。

## 快速开始

下面以 Swift 1.5 为例：

1. 准备 Windows 10 / 11、支持 CUDA 13 的 NVIDIA 驱动、12 GB 以上显存。内存建议 32 GB，第一次下载和转换模型需约 22 GB 空闲磁盘。
2. 下载懒人包，解压到简单路径，例如 `D:\ninfer-swift15`。
3. 双击 `启动.bat`，确认配置后按回车。第一次会下载、校验并转换模型；出现 `listening` 后即可使用。
4. 双击 `测试.bat` 检查回答。连接客户端时，按启动后生成的 `接入信息.txt` 填写接口和输出上限。

默认接口为 `http://127.0.0.1:8084/v1`，模型 ID 为 `qwen3.8-27b`。修改参数、模型准备和客户端接入见[使用指南](#swift-15-使用说明)；常见使用问题见[下文](#常见问题)，已知启动错误见[修复记录](CHANGELOG.md#启动错误与处理)。

## 本页导航

- [Swift 1.5 使用说明](#swift-15-使用说明)：启动、参数、客户端接入和模型准备。
- [测试结果与已知限制](#测试结果与已知限制)、[技术实现](#技术实现)、[源码与编译](#源码与编译)、[常见问题](#常见问题)、[来源与许可](#来源与许可)。
- Bonsai2 的完整说明：[12G README](README-12GB.md)、[8G README](README-8GB.md)。
- [更新与修复记录](CHANGELOG.md)：发布日期、构建识别和已知启动错误的处理。

## Swift 1.5 使用说明

### 运行前准备

- Windows 10 / 11，64 位。RTX 30 系、12 GB 以上显存已作为目标；RTX 40 系可运行，但没有实测。
- NVIDIA 驱动需支持 CUDA 13，现有包建议 580 以上。正常完整解压无需安装 CUDA 或 Python；VC 运行库 DLL 已随包携带。
- 内存建议 32 GB；第一次准备模型需约 22 GB 空闲磁盘。

把懒人包完整解压到简单路径，例如 `D:\ninfer-swift15`。`engine` 中的 DLL 必须与 `ninfer-serve.exe` 保持在同一目录。

### 启动和停止

1. 双击 `启动.bat`。默认选择 rk4v4、完整 MTP 输出头、200K 总上下文和自动输出。
2. 回车确认配置。首次使用会提示下载和转换模型，再按提示确认；下载可以断点续传。
3. 等待窗口出现 `listening`。首次遇到没有校准记录的显卡还会测量运行路线，因此比后续启动慢。
4. 双击 `测试.bat` 检查回答和速度。用完关闭启动窗口即可停止引擎。

同一时间只运行一个引擎。菜单中回车启动、C 重新配置、R 重新读取空闲显存、Q 退出。

```bat
启动.bat last
启动.bat dryrun
```

`last` 直接使用保存的配置；`dryrun` 只显示参数，不启动或下载。

### KVMem 与四种组合

### 按实际提示词分配 KVMem 窗口

启动时先按空闲显存确定总驻留容量 C；每次请求再准确计算系统提示、developer 指令和工具定义的完整开头 S。检索不包含 S，输出使用剩余容量。

| 总上下文 | 检索推荐范围 | 优先目标 | 输出预留 |
|---|---|---|---|
| 256K | 32K–36K | 36K | C − S − 检索，至少 8K |
| 128K | 16K–18K | 18K | C − S − 检索，至少 8K |
| 200K | 25K–28.125K | 28.125K | 同上 |

开头按完整渲染后的 token 数计算，Bonsai 按 64 token、Swift 按 128 token 对齐。不再固定预留 8K 开头，也不再把 KVMem 输出限制为 16K/32K。容量不足时先缩检索，低于推荐范围会提示；完整开头和 8K 输出都放不下时返回容量错误。实际生成仍受客户端请求长度、剩余总上下文、停止词以及模型结束影响；8K 是显存预留，不保证每次生成 8K。

Harness 在第一次请求前即可配置：`contextWindow` 填总上下文，`maxTokens` 使用启动器生成的数字。这是最短开头情况下的申请上限，实际请求的开头更长时，引擎会自动降低本轮输出预算。启动缩窗后重新生成接入信息；本机启动器在允许同步时同步 dsh。旧的手动 OUT/ANSWER、SYS/SINK 和 WINDOW 会迁移到自动分配。

8G 在 12GB 测试卡上也先扣除当前桌面/驱动占用，再按 8GB 预算计算。20K 是小容量测试或低空闲显存下的容量，67K 是假设整块 8GB 空闲时的理想预算；二者不能作为同一台电脑的固定上限。较小窗口可以运行，但超长历史召回可能变差。

支持的四种组合与历史解码测量如下；驻留容量以本次启动为准：

| 组合 | 总上下文 | 检索目标 | 输出 | 历史解码速度 |
|---|---|---|---|---|
| rk4v4 + 完整头 | 推荐 200K，最多 256K | 200K 时 28.125K；256K 时 36K | 剩余容量 | 约 43 token/秒 |
| rk4v4 + 精简头 | 最多 128K | 18K | 剩余容量 | 约 47 token/秒 |
| rk8v4 + 完整头 | 最多 128K | 18K | 剩余容量 | 历史测量慢 14–22% |
| rk8v4 + 精简头 | 最多 128K | 18K | 剩余容量 | — |

- **rk4v4**：对话记录的 K、V 都存 4 位。3060 上它反而比 rk8v4 **快 14–22%**（显存带宽是瓶颈，读得少就快），困惑度只高 0.10%。
- **精简头**：MTP 猜字时只在 13 万个常用词里猜，快约 10%，猜中率不变；代价是多占 340 MiB 显存，窗口变小。

### 修改参数

双击 `启动.bat` 后按 C，依次选择 KV 格式、MTP 输出头、看图、总上下文、单次输出、固定保留的开头和思考上限。回车保持当前值，选好后会写回 `设置.ini`。

也可以用记事本编辑 `设置.ini` 后重新启动：

| 设置项 | 作用 |
|---|---|
| `KV` | rk4v4 / rk8v4 |
| `HEAD` | full / lite |
| `CTX` | 总上下文，默认 204800 |
| `OUT` | auto，启动前生成 Harness 申请上限；实际预算随请求计算 |
| `SYS` | auto，完整系统／developer／工具定义的实际 token 数 |
| `THINK` | 思考上限，0 表示不限 |
| `POST_THINKING` | 思考结束后切换正文采样参数，默认 1；设 0 关闭 |
| `POST_THINKING_TEMP` / `POST_THINKING_TOP_P` / `POST_THINKING_TOP_K` | 正文的温度、top-p、top-k；留空使用引擎预设，温度默认为 0.2 |
| `POST_THINKING_SAMPLER` | 正文采样组合，例如 `temp=0.2,top_p=0.95` |
| `ADAPTIVE_MTP` | 自适应 MTP 草稿长度，默认 0，使用固定 3 token |
| `RECOVER_INVARIANT` | 内部不变量错误时尝试恢复请求，默认 1 |
| `VISION` | 是否启用看图 |
| `HOST` / `PORT` | 监听地址和端口 |
| `API_KEY` | 访问密码，默认留空 |
| `MODEL_ID` | 客户端使用的模型 ID |

窗口与自动输出会随空闲显存变化。开头与输出自动分配；启动后以生成的 `接入信息.txt` 为准。

推理控制设置已包含在 0.1.0 包中。`RECOVER_INVARIANT` 只处理引擎支持恢复的内部错误，不能保证所有错误都能恢复。

### 连接客户端

等待 `listening` 后，打开自动生成的 `接入信息.txt`。它包含当前配置和 DeepSeek Harness 的配置示例。

| 客户端字段 | 默认值或填写方式 |
|---|---|
| 接口类型 | OpenAI 兼容 / Chat Completions |
| Base URL | `http://127.0.0.1:8084/v1` |
| 模型 ID | `qwen3.8-27b` |
| API Key | 未设置密码时可填 `none`，部分客户端不允许为空 |
| contextWindow / 上下文长度 | 与 `CTX` 一致，默认 204800 |
| maxTokens / 最大输出 | 按本次 `接入信息.txt` 的数值填写 |

接口也支持 Anthropic Messages 和 OpenAI Responses。思考内容在 `reasoning_content` 字段；请求中 `enable_thinking: false` 可以关闭思考。支持的思考档位为 low / medium / xhigh，加上关闭思考。

一次处理一个请求，多个请求会排队。需要局域网访问时，将 `HOST` 改为 `0.0.0.0` 并设置 `API_KEY`；接入信息中会列出可用地址。

修改 dsh 的配置后，完全退出程序和托盘图标，再重新打开，避免它把内存中的旧配置写回文件。

0.1.0 对客户端发送的较大输出上限按服务端配置封顶；中途压缩、编辑历史或分叉的 KVMem 请求回退到重新预填，因此可能比直接续写慢。多工具 `required/any` 按 `Auto` 处理，不能保证一定调用工具；接受 `strict:true` 的入口也不提供严格 JSON Schema 约束。Swift Responses 入口仍拒绝 `strict:true`。具体覆盖范围见[三项兼容修改与限制](CHANGELOG.md#2026-10-04三项智能体兼容修改与-bonsai2-12g-源码公开)。

### 模型下载和转换

模型来源：[魔搭社区 Swift-1.5-Qwen3.8-27B-GSQ-RCO-NInfer](https://www.modelscope.cn/models/fyb423/Swift-1.5-Qwen3.8-27B-GSQ-RCO-NInfer)。

- 原始下载文件：`swift15_iq2s_mtp.ninfer`，10,257,632,000 字节。
- 启动器使用包内 `patch` 转换成 `swift15_iq2_s_mtpq4.ninfer`，10,155,600,640 字节。
- 只修改 MTP 草稿头的 7 个张量，其余模型对象保持原样；下载和转换后都进行 SHA256 校验。
- 转换使用 PowerShell，无需安装 Python。校验通过后，启动器会删除原始下载文件。
- 也可以自行下载原始文件，按原名放入 `model` 目录，再启动完成转换。

文件 hash 和自行生成补丁的方法见[编译与开发](#模型补丁与转换工具)。已有转换完成的模型可继续使用，升级引擎无需重新下载模型。

### 包内文件

| 路径 | 内容 |
|---|---|
| `启动.bat` / `测试.bat` | 启动和测试入口 |
| `设置.ini` | 用户配置 |
| `engine/` | EXE 与运行所需 DLL |
| `launcher/` | 启动器和测试脚本 |
| `patch/` | 模型转换的数据补丁 |
| `model/` | 初始为空，下载与转换后的模型 |
| `版本信息.txt` | 新修复包的构建识别信息 |
| `SHA256SUMS.txt` / `verify-kit-manifest.ps1` | 新版包的核心文件哈希、DLL 完整性和中文文件名检查 |
| `verify-arch-engine.ps1` | 新版包的引擎验收脚本，包含短提示、超窗正负对照与多轮复用 |
| `logs/` / `接入信息.txt` | 运行时生成的日志和客户端说明 |

包里的 `使用说明.txt` 可离线查阅；速度与长文召回范围见[测试结果与已知限制](#测试结果与已知限制)。

上述校验和验收工具已随 0.1.0 源码及懒人包提供。完整包解压后可运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\verify-kit-manifest.ps1
powershell -ExecutionPolicy Bypass -File .\verify-arch-engine.ps1
```

第一项检查文件；第二项会启动测试引擎并占用显存，运行前关闭其他模型。模型默认从包内 `model` 目录读取，也可用 `-ModelPath` 指定。`SHA256SUMS.txt` 和校验脚本对应指定发布构建；自行编译或修改脚本重新打包时，需要一起更新清单与脚本内的预期哈希。

## 测试结果与已知限制

这里按版本保存已有测量和验证范围。不同工具、模型格式、提示词和输出长度的速度不能直接横向比较；文档整理没有重新测量这些历史结果。256K 是可配置的总上下文，KVMem 每轮只读入选中的历史，不能据此保证所有内容都能精确召回。

### 质量（困惑度，越低越好）

同一台机器、同一个测试工具（引擎自带的 `ninfer-perplexity`，语料 `perplexity-1m`，上下文 4096）：

| 领域 | 12G 版 Swift-Bonsai-2 | **Swift 1.5 IQ2_S** |
|---|---|---|
| 中文维基 | 7.894 | **5.443** |
| 英文长文 PG-19 | 8.656 | **7.439** |
| 英文 WikiText | 7.827 | **6.321** |
| 代码 | 1.860 | **1.716** |
| **总体** | 5.628 | **4.587**（低 18.5%） |

### 速度（RTX 3060 12GB）

| 指标 | 数值 |
|---|---|
| 生成速度（引擎自带 refbench，MTP 投机解码） | **约 43–45 token/秒** |
| 生成速度，MTP 输出头选"精简" | **约 47 token/秒**（+10.5%） |
| 读入速度（4K / 24K 上下文） | 514 / 453 token/秒 |
| 19 万 token 第一次读入 | 约 9 分钟 |
| 25 万 token 第一次读入 | 约 12 分钟（约 365 token/秒） |
| 长对话接着问（19 万 token 之后） | 每轮约 **1.3–1.4 秒**开始回答 |

- 和 12G 版用同一个工具（refbench、32K、都用 rk8v4）对比过：短回答生成慢约 8%，长上下文慢约 20%，读入慢约 35%（2-bit 权重计算量更大）。之后默认换成 rk4v4，生成又快了 14–22%。换来的是质量明显更好、上下文更长。
- 12G 版介绍里的"约 53 token/秒"是用另一个工具（`ninfer_bench`）测的，和这里的数字不能直接比。
- 速度测试都是开关交替测几轮，排除显卡发热降频的影响。

### 长上下文

| 场景 | 结果 |
|---|---|
| 12 万 token 长文藏一句话 | **找到**；多轮追问、切到别的对话再切回都正常 |
| 19 万 token 长文藏句、工具调用、多轮复用 | 单次输出 16K 时 5 次全部通过；输出调到 23K（检索区变小）时，工具那一项没过 |
| 25 万 token 长文藏一句话 | **没找到**：暗号所在的那一块没被挑进显存（见本页的已知限制） |
| 6 万 token 对话，超出显存窗口以后接着问 | 每轮 2–4 秒（改之前每轮约 140 秒） |

### 测试环境

| 项 | 配置 |
|---|---|
| 显卡 | RTX 3060 12GB（28 SM，显存带宽 360 GB/s） |
| CPU | Ryzen 7 3700X |
| 内存 | 32 GB（Host KV：200K 上下文预留 8 GB，256K 预留 10 GB） |
| 系统 | Windows 11 |
| 前端 | DeepSeek Harness 桌面版（其他 OpenAI 兼容客户端也能用） |

### 已知限制

- **25 万 token 藏句没找到**：按块打分时，暗号所在那一块得分偏低，没被挑进显存。需要精确查找时，材料尽量控制在 12–19 万以内；贴很长的材料时，先发材料，再单独发一条问题，召回会好很多。
- **第一次读入慢**：19 万约 9 分钟，25 万约 12 分钟。之后接着问很快。
- **一次只处理一个请求**：同时发多个会排队。
- **单次输出（含思考）有上限**：自动时 3060 上约 23–28K，超过会被截断，写很长的东西要分几步。
- **显存大于 12 GB 的卡**：窗口有上限，多出来的显存暂时用不上。
- **只支持 RTX 30 / 40 系**：引擎按 sm_86 编译。RTX 50 系请用 Ryan-gsq 分支的官方预编译包。

### 测试方法

- 速度用引擎项目自带的 refbench，困惑度用 `ninfer-perplexity`。开关交替测，排除显卡过热降频的影响。
- 长上下文用藏句找回、多轮追问、工具调用、切对话再切回测试，每次改动后都回归。
- 懒人包实测过下载（含断点续传）、校验、补丁转换、真机启动和 `测试.bat`。

## 技术实现

下列性能数字对应原有测试，不表示每次更新都会再次获得同等提升。按日期发生的修复和发布状态见[更新记录](CHANGELOG.md)。

### 1. 让新引擎在 RTX 30 系上跑

Swift 1.5 的模型格式（GGUF 量化块，IQ2_S / IQ3_XXS 等十几种）只有 ninfer-all 这条新分支能读，它的预编译包只支持 RTX 50 系。移植时：

- 按 sm_86 编译，补齐 cuBLAS 等依赖；
- **精简引擎**（编译开关 `NINFER_SLIM_3060`）：只保留单卡、单路要用的注意力内核，显存和编译时间都省下来；
- **SM 数运行时自动识别**：同一个程序在 3060（28 SM）和更大的卡上都能用，不用重新编译。

### 2. KVMem：12 GB 显存跑 256K 上下文

- 底子是 qzshch/ninfer-kvmem 在新引擎结构上的实现：完整对话放内存（Host KV），显存块表里用"空洞"表示不在显存的页，位置保持原样；
- **Windows 上 Host KV 改用可分页内存**：原来用锁页内存，Windows 上分不出 8 GB；
- 检索按 kv9 的做法完整移植：按消息分段做查询，挑块时对准换行，打分时排除开头、保护块和最近块，CPU 打分矩阵化、多线程；
- 和 MTP 投机解码、看图都兼容。

### 3. 输出留在显存、超窗后只重算新内容

- 单次输出整段预留在显存里，不会被挤出去；开头（系统提示）固定保留；
- 历史超出窗口以后，多轮对话只重算新内容。6 万 token 的对话，接着问每轮从约 140 秒降到 **2–4 秒**；
- 多个对话来回切时，长对话的接续点不会被挤掉。

### 4. 省显存（每一项都换成更大的窗口）

| 改动 | 省下 | 代价 |
|---|---|---|
| 词嵌入放内存（`--embedding-host`） | 约 360 MiB | 无，速度不变 |
| 视觉权重放内存（overlay） | 约 280 MiB | 第一张图约 1.7 秒 |
| MTP 草稿头压成 Q4（用补丁改模型文件） | 约 95 MiB | 无，速度和猜中率在波动范围内 |
| rk4v4 | 32K 上下文省约 260 MiB | 困惑度 +0.10% |

### 5. 提速

- 注意力跳过不在显存的空洞页；
- 页面从内存搬上显卡时批量拷贝（gather），按碎片程度自动选用；
- 精简 MTP 输出头做成可选项（+10.5%）。

### 6. 启动器

- 按启动时的空闲显存自动算窗口，整卡留约 150–200 MiB；估多了引擎会报真实可用量，启动器自动缩小窗口重试；
- 单次输出可以选"自动"，按窗口大小分配；
- 先显示上次配置，回车直接启动；按 C 一步步重新选，每一步都写着推荐值和上限。

### 当前源码中的构建支持

| 改动 | 内容 |
|---|---|
| KVMem | 移植 qzshch/ninfer-kvmem 的空洞块表、设备放置事务、稀疏工作集、滚动预填窗口、解码环形窗口和检索机制 |
| 3060 精简 | `NINFER_SLIM_3060` 保留单卡、单路、H24 注意力和 int8 / rk8v4 / rk4v4 内核 |
| 内存和显存 | Windows Host KV 用可分页内存；词嵌入放锁页内存；开头和输出按环境变量预留 |
| kv9 检索 | 按消息分段查询、对准换行、排除开头/保护块/最近块、CPU 打分矩阵化多线程；默认查询行数 1024 |
| 注意力和换入 | 跳过空洞页；页面换入按碎片程度自动选批量 gather |
| SM 自适应 | GDN 路线表运行时按真实 SM 数选择；rope 容量 = 6 × SM |
| Windows 显存预算 | 引擎读取 CUDA 和 NVML 空闲显存，NVML 可用时取两者较小值；`NINFER_FREE_VRAM_MIB` 可进一步限制预算 |
| KVMem 内容打分 | 包含单段查询和达到提示窗口边界的选择路径，日志用 `kvmem_score: SELECT` 标识 |
| 连接保活 | Windows TCP 保活为 10 秒 / 3 秒，HTTP 连接保活 120 秒，读写超时 300 秒 |

后三项对应 2026-10-04 同步的源码。它们的修复背景和发布状态见[更新记录](CHANGELOG.md#2026-10-04swift-15-q2s-源码与启动器同步)；原有性能表保持此前测量结果。

## 源码与编译

Swift 1.5 的完整引擎源码在 [`swift15/engine/`](swift15/engine/)，包含 CMake 工程、实现、应用、测试和第三方源码。启动器和设置在 [`swift15/oneclick/`](swift15/oneclick/)，构建脚本在 [`swift15/scripts/`](swift15/scripts/)。目录索引见 [swift15/README.md](swift15/README.md)。

Bonsai2 12G 的完整源码、构建脚本和启动器已公开在 [`bonsai2-12g/`](bonsai2-12g/README.md)，从源码编译见 [12G README](README-12GB.md#源码与编译)。两套引擎分别对应不同模型格式。8G 方案的专用改造源码尚未整理公开。

### 从源码编译

#### 1. 下载源码

```bash
git clone https://github.com/5258MF/ninfer-rtx3060-27b.git
cd ninfer-rtx3060-27b
```

也可以在 GitHub 上点击 Code → Download ZIP。引擎源码位于 `swift15/engine/`，上游基线是 Ryan-gsq 分支 **b06908b**，此前的改动均已合入。

#### 2. 环境（Windows）

- CUDA 13.x（要带 cuBLAS；我们用的是 13.3）
- MSVC 2022（C++ 桌面开发）、CMake、Ninja
- FFmpeg 开发包（看图用）、zlib、libcurl（我们用 conda 里的）

#### 3. 编译

改好 `swift15/scripts/build-sm86.bat` 开头的 CUDA、MSVC 和依赖路径，然后在仓库根目录运行：

```bat
swift15\scripts\build-sm86.bat
```

脚本默认使用仓库里的 `swift15/engine/`，输出到 `swift15/build/`；源码和构建路径使用纯英文路径。产物是 `swift15/build/apps/ninfer-serve.exe`。把它和运行要用的 DLL（cudart、cublas、cublasLt、FFmpeg、zlib、libcurl、libssh2、libcrypto、zstd 等）放到懒人包的 `engine\` 目录。

- 只编译 sm_86。RTX 30 系能跑，RTX 40 系（sm_89）二进制兼容也能跑；RTX 50 系不行，请用上游的预编译包。
- SM 数在运行时自动识别，不同型号不用重新编译。
- 改动头文件以后，完整重新编译可能要近 1 小时。

#### 4. 必须加的启动参数

单卡 12 GB 上，`--device-state-slots 0 --prefill-chunk 512` 必须加，否则显存不够。完整参数由启动器生成，大致是：

```
ninfer-serve swift15_iq2_s_mtpq4.ninfer --model-id qwen3.8-27b
  --kv-dtype rk4v4 --gdn-state-fp16 --spec mtp --draft-tokens 3
  --max-context 204800 --kv-capacity auto --kvmem-window-pages <按空闲显存算>
  --host-kv-mib 8192 --device-state-slots 0 --prefill-chunk 512
  --cuda-graph-allowance-mib 144 --default-max-tokens <输出>
  --max-private-continuations 4 --embedding-host
  [--lm-head-draft] [--vision --vision-residency overlay --vision-max-merged 4096]
```

环境变量：`NINFER_KVMEM_SINK_PAGES`（开头保留页数）、`NINFER_KVMEM_GEN_RESERVE_PAGES`（输出预留页数）、`NINFER_KVMEM_LONG_REUSE=1`。Windows 上开 KVMem 时 Host KV 自动用可分页内存（`NINFER_HOST_KV_PAGEABLE=0/1` 可以强制）。1 页 = 64 token。这些值由[启动器源码](swift15/oneclick/launcher/launch.ps1)计算。

### 模型补丁与转换工具

懒人包不放模型，第一次启动时从魔搭下载 `swift15_iq2s_mtp.ninfer`（10,257,632,000 字节，SHA256 `2df7259e93cbbb972183966e40ff23392e66a8ce4fc84748d08849bc6182045e`），再用补丁转成 `swift15_iq2_s_mtpq4.ninfer`（10,155,600,640 字节，SHA256 `1941ef5f4bd4be61938521236d0e499c5e6d69ff69b4fbcfe83527f7fd9b3bdd`）。

自己生成模型补丁（在 `swift15/engine/` 里运行，要 `pip install gguf numpy torch`）：

```bash
python -X utf8 -m tools.mtpq4 swift15_iq2s_mtp.ninfer swift15_iq2_s_mtpq4.ninfer
python -X utf8 -m tools.mkpatch swift15_iq2s_mtp.ninfer swift15_iq2_s_mtpq4.ninfer patch_out
```

两个脚本已经放在引擎的 `tools/` 目录，会使用同一源码树中的 `tools.artifact`。

`mtpq4.py` 只改 MTP 的 7 个张量：注意力 query/gate（Q4）、key/value（Q8，引擎的 linear_pair 只收 Q8）、注意力 output（Q4）、MLP gate+up 和 down（Q4）；`input_projection` 没有对应形状的 Q4 内核，保留 Q6_K。量化用项目自带的 `quantize_matrix_mse`。其余对象逐字节复制。

### 升级现有包

重新编译后替换包内 `engine/ninfer-serve.exe`，保留完整依赖 DLL。已经下载和转换的模型可以继续使用；修复所需版本见[更新记录](CHANGELOG.md)。

## 常见问题

### 显存或主机内存分配失败

| 情况 | 处理 |
|---|---|
| 显存分配失败或窗口放不下 | 关闭其他占显存程序；按 R 重新读空闲显存，或按 C 调小单次输出 |
| 运行后突然很慢 | 检查是否显存被其他程序占用，释放后重新启动 |
| `cudaMallocHost` / `bad allocation` | 检查主机内存和其他模型占用，可调小总上下文；缩显存窗口不能覆盖所有主机锁页失败 |
| 已有模型运行或端口占用 | 关闭原模型，或修改 `设置.ini` 的 `PORT` |

启动器只处理它识别的容量报错，不能保证所有分配失败都自动恢复。输入很长的材料时，也需要考虑[长文召回范围](#测试结果与已知限制)。

### 下载、转换与客户端

- 下载中断：重新启动，已有 `.part` 文件会继续下载。
- 校验失败：检查模型文件是否完整，按包内说明重新下载。
- 补丁缺失：重新完整解压 `patch` 目录；现成的转换后模型可以继续使用。
- 客户端输出截断：最大输出按本次生成的 `接入信息.txt` 填写；思考也占输出预算。
- dsh 改完不生效：退出程序与托盘图标后再打开，避免旧配置写回。
- 窗口闪退：在包目录地址栏输入 `cmd`，再运行 `启动.bat` 查看具体错误。

引擎日志在包内 `logs` 目录。分享错误信息时，去掉 API Key 等私人信息。

DLL 缺失、KV 容量曲线越界或首次显卡校准失败时，见[启动错误与处理](CHANGELOG.md#启动错误与处理)。

## 来源与许可

代码许可与模型许可分别适用。Swift 1.5 引擎代码的 Apache-2.0 文本见 [swift15/LICENSE](swift15/LICENSE)，源文件保留上游许可证和贡献者信息。

- **Swift-1.5 Qwen3.8-27B GSQ-RCO**：UkisAI（Swift Open License v1.0，允许署名再分发；年收入超过 100 万美元的商用需要另外和 UkisAI 谈）；基座 Qwen3.8-27B（Apache-2.0）。
- **.ninfer 格式模型**：魔搭社区 fyb423 转换发布（[Swift-1.5-Qwen3.8-27B-GSQ-RCO-NInfer](https://www.modelscope.cn/models/fyb423/Swift-1.5-Qwen3.8-27B-GSQ-RCO-NInfer)）。
  - 本项目对模型的改动（Swift Open License 4(b) 要求的变更说明）：只把 MTP 草稿头的 7 个张量从 Q6_K 重新量化成 Q4（query/gate、output、MLP）和 Q8（key/value），其余权重逐字节不变。懒人包里只放补丁，不放模型本身，模型由用户从魔搭下载。
- **引擎**：Ryan-gsq/ninfer-16g-5070ti-5080-5090-qwen3.8-27b-gsq-rco（Apache-2.0），它是 iamwavecut/ninfer-all 的分支，最早来自 Neroued/ninfer。
- **KVMem**：qzshch/ninfer-kvmem（Apache-2.0）的新引擎实现；kvmem-llama.cpp 项目和它的论文。
- 3090 分支和其他 3060 用户公开的调优记录，也提供了很多参考。
