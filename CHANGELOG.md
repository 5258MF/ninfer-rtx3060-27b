# 更新记录

[项目首页与下载](README.md#下载) · [故障排查](#启动错误与处理) · [测试结果](README.md#测试结果与已知限制)

按版本和日期记录已发生的改动。源码更新与网盘包更新分别标明，避免把源码中的修复误认为已经进入旧下载包。


## 2026-10-06：通用启动器纠正与评分／性能报告（Swift 0.1.4 / Bonsai2 0.1.5，Release 再次撤回）

Swift 0.1.4、Bonsai2 0.1.5 源码 Release 按用户要求再次撤回，发布页面及附件已移除。Git 标签和 main 源码保留；这两版仍有待修正问题，暂不重新发布。历史 Swift 0.1.3、Bonsai2 0.1.4 Release 及网盘包保持不变。

- 同步四包修正：自动 KVMem 只定驻留预算和有限 API 上界，移除旧固定 SYS、固定 36K／平分和 32K 输出封顶；`default-max-tokens` 在此引擎中也限制显式请求。完整前缀、检索与输出交给引擎逐请求分配。IQ3／8G 的测试用 16K API 限制取消。保留命令行参数读取、CPU／Graph 控制和旧 KV 页回收修复。
- 从公开源码移除四包的 DSH 测试入口、预设和客户端脚本。B 机原会话保留在本地归档；通用包只提供兼容 API，不绑定某个框架或压缩策略。个人配置、会话、原始请求和运行日志未上传。
- 主 README 更新 IQ3XXS PPL 4.495637819904494；Q6/Q4 两份原报告共 261167 评分 token、124 窗口，主模型评分一致。CausalScoring 不验证 MTP 草稿或视觉。19 份原始 PPL 报告按原哈希保存，早期 custom 语料不与 quick 混算；IQ2 旧 PPL 未找到原报告，保留历史参考标签。
- 用 2026-10-05 同口径数据替换主展示速度表：60 个文本／代码正式样本，以及 90 个正式 2K/8K/16K prefill 样本。保留测试驻留窗口、Graph、MTP、缓存计数、TTFT 和异常记录。旧 refbench／ninfer_bench 优化数字移作历史记录，不据不同输入／工具宣称新代码提速。公开数据和哈希见 `benchmarks/20261005/`。
- B 机最新修正的 12 个短请求包括 8582-token 完整前缀，未观察到 OOM；它们不是 32K／64K 长输出验收。新默认窗口与旧速度测试窗口不同，未把旧高窗口成绩当成当前默认保证。

English: Framework-neutral launchers, request-time allocation, archived IQ3 PPL and decode/prefill tables remain in main. Test-only DSH wrappers are removed. Swift 0.1.4 and Bonsai2 0.1.5 source Releases have been withdrawn again at the user’s request because issues remain. Release pages and assets are removed; tags and main are retained pending fixes. Historical Releases and cloud ZIPs are unchanged.

## 2026-10-06：KV 换页与启动器修复（Swift 0.1.4 / Bonsai2 0.1.5，Release 已撤回）

**本次更新 GitHub 源码与两套独立源码 Release。网盘懒人包及下载链接保持现有发布状态；源码版本不代表旧 ZIP 已包含修复。**

- Bonsai2 修复 KVMem 换入时不能回收旧设备页的问题。非当前成员页必须没有活跃引用、写入引用或源锁定；已有有效且最新 Host 副本时直接通过原有保护释放设备副本，避免重复 `host.prepare()` 被拒绝。无有效副本仍走原 D2H 路径。开关为 `NINFER_KVMEM_ORPHAN_EVICT_FIX=1`，8G 的 kvrk4 启动器启用，其余构建默认保持关闭。8G / 12G 共用源码，实机修复验收针对 rk4 构建。
- 四个配置补齐 CPU 视觉、CPU 线程与 CUDA Graph 开关处理。Swift 使用 `--vision-residency cpu`，Bonsai 使用 `NINFER_VISION_CPU=1`；关闭 Graph 时去除 allowance 并传 `--no-cuda-graph`。保留完整系统/developer/工具前缀按请求分配、输出使用剩余容量且最低 8K 的现有逻辑；没有用 B 机旧启动器覆盖回固定开头或 32K 封顶算法，也保留了已有分配回归测试。
- 8G rk4 默认逻辑上下文 128K、关闭 Graph；KVMem 驻留上限 36K，仍按空闲显存下调，并非强制分配 36K。IQ3XXS 开发配置为 33K 驻留、128K 逻辑上下文、API 默认 16K，实际前缀/检索/输出在请求时确定。API 申请上限不等于实际可生成长度。
- 新增独立的 DSH 精简/普通入口。普通预设的压缩预留为 8192 token，摘要上限为 4096；精简入口加载 DSH 自带 minimal 预设。依赖通过 `DSH_MODULES_DIR` 和 PATH/`DSH_NODE_EXE` 指定，使用包内独立配置与会话，不修改全局 DSH 配置。入口需要已有 Python、Node 和 DSH，不是随源码附带这些软件。
- IQ3XXS 启动器源码放在 `swift15/oneclick-iq3xxs/`，与 IQ2S 共用引擎。附测量配置及路径覆盖选项，不附模型、EXE、DLL 或独立懒人包。只允许已验证的 rk4/noGraph 默认档启动，其余容量记录可预览。

交接验收：旧页修复关闭时定向回归出现 HTTP 500，开启后 15 次请求完成，并释放 1169 页；另有 11 项模拟存储保护检查。原真实 DSH 精简长链完成约 11 万输入 token、34 次工具调用。四套部署启动器的短工具链完成读取、写入、回读。新 36K 自动档未完成长压力与连续 16K 输出验收，不能把这些记录解释为所有窗口和负载均已验证。本次 A 机检查通过：11 个相关 PowerShell 脚本语法、8 组 Graph 开关 dryrun、4 组 CPU/Graph 参数检查、8 组 DSH 隔离配置生成；测试使用临时夹具，没有重新加载 GPU。另修复公共脚本加载时的命令行参数保留、Swift dryrun 仍弹菜单，以及新运行选项未进入配置读取白名单的问题。

English: Bonsai2 source 0.1.5 includes the opt-in old-KV-page eviction fix, enabled by the 8G kvrk4 launcher. Swift source 0.1.4 and all four launcher profiles add CPU vision/thread and CUDA Graph controls plus isolated DSH entry points. Existing request-time prefix/retrieval/output allocation and its tests are preserved. The IQ3XXS development launcher shares the Swift engine; no portable IQ3XXS ZIP is published. Existing cloud packages are unchanged. Targeted GPU and DSH evidence comes from the recorded handoff; the new 36K profile has not completed long-output or full stress qualification.

## 2026-10-05：Bonsai2 共用源码目录更名

Bonsai2 8G 与 12G 共用同一份引擎源码；仓库目录由 `bonsai2-12g/` 改为 `bonsai2/`，目录内 `oneclick-8g/` 和 `oneclick/` 分别保留 8G、12G 启动器。此次只调整命名和文档链接，不改变引擎代码或懒人包参数。已发布的 `bonsai2-12g-v0.1.3` 标签保留原名，作为历史版本标识。

## 2026-10-05：Bonsai2 12G 启动器目录明确命名

Bonsai2 8G / 12G 继续共用一个引擎源码目录。12G 启动器源码由 bonsai2/oneclick/ 更名为 bonsai2/oneclick-12g/，与 oneclick-8g/ 对称；只整理源码目录名和文档链接，不修改启动器参数或网盘懒人包。Bonsai2 共用源码版本升为 0.1.4，新的源码 Release 附件只包含 bonsai2/。8G、12G 懒人包仍为 0.1.3。

English: The 12G launcher source now lives in bonsai2/oneclick-12g/, alongside oneclick-8g/. This is a source-tree and documentation rename only; the 0.1.3 packages are unchanged. Shared Bonsai2 source version 0.1.4 is released separately.

## 0.1.3 - 2026-10-05：Swift 1.5 IQ2S / IQ3XXS 共用源码

**状态：Swift 1.5 共用源码版本为 0.1.3；当前 IQ2S 懒人包仍为 0.1.2，IQ3XXS 懒人包尚未发布。**

- Swift 1.5 的 IQ2_S 与 IQ3_XXS 使用同一份 swift15/engine/ 源码。引擎已有两种量化格式的 NInfer 映射和对应 GPU 内核；不需要复制源码目录。当前 swift15/oneclick-iq2s/ 仍是 IQ2S 启动配置。 对齐 Bonsai2 的目录方式后，现有启动器目录从 swift15/oneclick/ 更名为 swift15/oneclick-iq2s/；后续可并列添加 IQ3XXS 启动器目录。
- 0.1.3 将 CPU 视觉编码适配合入这套共用引擎：可选 OpenBLAS、自动线程选择与缓存；OpenBLAS 不可用时回退到参考 CPU 实现。该路径对两种量化格式共用。
- B 机记录：Swift IQ2S / IQ3XXS 看图功能测试、CPU 后端 7 项单测及相关 Windows 构建通过。没有据此声称所有图片、CPU 或长上下文配置均完成验收。
- [Swift 1.5 IQ2S / IQ3XXS 共用源码 Release 0.1.3](https://github.com/5258MF/ninfer-rtx3060-27b/releases/tag/swift15-v0.1.3) 附件只含 swift15/ 源码目录。IQ3XXS 懒人包待后续单独制作；现有网盘下载表不变。

English: Swift IQ2_S and IQ3_XXS share the swift15/engine/ source and quantization kernels. The existing launcher source is now named oneclick-iq2s/; no IQ3XXS package is included yet. Source 0.1.3 adds CPU vision encoding with optional OpenBLAS, cached thread selection, and a CPU fallback. The current IQ2S package remains 0.1.2.

## 0.1.3 / Swift 0.1.2 - 2026-10-05

**状态：Bonsai2 8G／12G 懒人包为 0.1.3，Swift 1.5 Q2S 为 0.1.2。三款懒人包现共用同一组百度、夸克网盘分享链接，文件名和提取码见项目首页下载表。**

- 已移除整仓快照的 `v0.1.0` Release 页面；Git 标签仍保留为历史快照，模型源码请使用各自独立的 Release。

- Qwen 3.8 27B 模型库已合并到同一网盘文件夹；README 中已更新最新百度、夸克链接和提取码。

- 三套方案按本次完整渲染的系统、developer 和工具定义计算固定开头；Bonsai 对齐 64 token，Swift 对齐 128 token。256K 检索目标 32K–36K、优先 36K；128K 为 16K–18K、优先 18K，检索不包含固定开头。
- 输出使用剩余驻留容量，最低预留 8K，取消 KVMem 的固定 16K／32K 上限。空间不足先缩检索；完整开头加 8K 输出不能容纳时返回容量错误。自然的总上下文、客户端长度和停止条件仍然生效。
- 启动前给 Harness 生成稳定的 contextWindow 和最大申请上限；每次请求再降低实际输出预算。六个本机／懒人包启动器同步修改，旧的固定开头、输出和手动历史上限迁移为自动分配；缩窗后重算接入参数。
- Swift 共享前缀摘要不匹配时曾误索引私有缓存目录。共享槽数大于私有槽数时会触发 Windows 0xc0000005。现在只对私有条目查询私有会话；新增共享 3 槽／私有 1 槽的失配回归。原来失败的 128K 请求序列重新通过。
- Swift 图片完整组按本轮开头和历史预算校验，最小检索预算也优先保留完整开头；不再使用固定两页开头做图片预算。
- 三个增量构建通过；分配数学和嵌套作用域、完整资源管理器测试、714 项启动器容量／缩窗／语法检查通过。Bonsai 四种配置与 Swift 的 128K／256K 覆盖 Chat、Messages、Responses、工具参数、长系统提示、容量拒绝及拒绝后恢复。
- dsh 0.2.0-rc.2 自带的 standard／minimal 预设均完成真实读文件、写入 42、再读回核对的三次工具往返，覆盖 Bonsai 8G／12G 和 Swift。测试通过原生注册器显式绑定预设，未修改用户配置或使用私人提示。Bonsai 12G 同 80K 驻留容量下，输出预留从两种模式的 32K 变为约 37.6K／43.6K；短任务耗时约 13／9.3 秒，新旧基本相同，没有测出提速。
- 8G 在 20K 驻留容量下，两种 dsh 预设可以运行并保留 8K 输出，检索分别约 5.6K／11.6K，明显低于 256K 推荐范围。另一项 11K 系统提示＋30K 重复正文、仅 1K 检索的标记测试答错，不能宣称小窗口超长召回全部通过。没有真实 8GB 硬件、完整最大长度生成或新困惑度验收；旧 KV 单测的历史失败没有算作本轮通过。

English: Request-based allocation preserves the complete instruction/tool prefix, targets 36K retrieval at 256K or 18K at 128K, and gives the remainder to output with an 8K reserve. Harness limits are available before the first request. A Swift shared/private catalog indexing defect is fixed. Shipped dsh standard/minimal presets completed real tool round trips on all three configurations; output headroom improved at the same resident capacity, with no demonstrated short-task speedup. Small-window long-history recall remains limited.

- 新上传的 `swift15_iq3xxs_mtp.ninfer` 为 NInfer v3 文件；12G Bonsai 引擎当前只接受 NInfer v2，实际启动在读取文件头阶段以 `artifact magic is not NInfer v2` 退出，尚未加载权重或测试显存。该文件不能直接搭配当前 Bonsai 版懒人包。

## 0.1.2 - 2026-10-04

**状态：Bonsai2 8G / 12G 源码与本机懒人包更新；网盘仍为旧包，等待重新上传。Swift 1.5 保持 0.1.1。**

- KVMem 的 FP32 块和计数索引、BF16 查询暂存迁到锁页内存；保持 GPU 按原顺序追加索引，CPU 用 AVX2/FMA 分块打分，支持标量回退。快照恢复和 CUDA Graph 重放覆盖原布局。
- CPU 打分在本段索引追加前完成，使用预填开始时的历史计数，避免当前段或部分尾块进入错误统计域；召回掩码、消息查询和段融合保留原规则。
- 窗口按 Swift 1.5 默认逻辑分配：先预算总容量，优先留 36K 历史，自动回答封顶 16K（8G rk8v4）/32K（其他 Bonsai 模式），余量全给历史。取消固定历史上限与旧自动回答预算推导，手动回答保持不变。
- 公开 8G 启动器和验收源码，与 12G 共用 `engine/`；构建脚本支持 `NINFER_RK4_SM86=ON`，RK4 使用独立输出目录。
- 两个 CUDA 构建通过；FP64 打分 oracle 与 GPU/映射主机索引精确对照、部分块、快照恢复、Graph 重放单测通过；四种 KV/模型功能验收通过，包括工具往返、输出封顶、Messages/Responses 和图片。
- 同容量整卡采样节省约 401–404 MiB；短请求输出一致，解码速度基本持平，短提示词读入略慢。扩容与测试边界见 [8G](README-8GB.md)、[12G](README-12GB.md) 说明。
- 12G 在 112K 驻留容量（80K 历史 + 32K 输出预留）下，252,552 token 的三处藏针及两轮续问通过。8G CPU 在 56K 驻留容量下首次提问 A 时答成 C；原 GPU 路径同容量、同输入也把 A 答成 C（B 同样漏召回，C 正确）。至少首次错误在原路径中也重现，不能宣称该项通过。
- Swift 窗口分配回归 788 项、最终命令/环境变量对照 18 项通过；扩大窗口的四种 KV 配置均通过工具、双 API、看图与至少 256 token 连续解码。8G 同容量看图对照为 GPU 7721 / CPU 7319 MiB；单次扩大窗口整卡采样曾出现 8048 MiB，不能按该配置保证真实 8GB 看图峰值。
- 本轮未重测完整 16K/32K 长输出、困惑度或真实 8GB 硬件。此前完整旧 KV 单测的三项历史断言失败没有算作通过。

English: Bonsai2 moves its FP32 index and BF16 query stash to host RAM and uses blocked CPU scoring, saving about 400 MiB at the same capacity. Allocation now follows Swift's default policy: budget the whole resident window, cap auto output, and assign the remainder to history. Explicit answer limits stay unchanged. Small-window functionality and the 12G 252K-token retrieval test passed; 8G long-retrieval comparison is reported separately.

## 0.1.1 - 2026-10-04

**状态：0.1.1 源码已按模型分别发布 Release；本机新懒人包已验证，网盘待用户重新上传。0.1.0 标签、各模型独立源码 Release 和现有网盘链接保留原快照。**

- 根目录、Swift 1.5 和 Bonsai2 12G 的 VERSION 均升级为 0.1.1。
- [Swift 1.5 Q2S 0.1.1 源码 Release](https://github.com/5258MF/ninfer-rtx3060-27b/releases/tag/swift15-v0.1.1) 与 [Bonsai2 12G 0.1.1 源码 Release](https://github.com/5258MF/ninfer-rtx3060-27b/releases/tag/bonsai2-12g-v0.1.1) 分别提供对应模型的完整源码附件；不创建新的合并源码 Release。
- 软件版本包含启动器、验收和发布资料；C++ 与引擎二进制未改变，版本递增不代表重新量化或性能变化。

- 根据 listening 日志判断启动完成，替代 120 秒运行时长阈值；仅启动期容量错误触发缩窗，运行中退出采用有上限的重启。
- 缩窗后按实际输出上限更新客户端接入信息文字与 YAML；本机启动器在允许同步时同步 dsh，保存配置不写入临时缩窗结果。必要时联动限制思考预算。
- 启动失败返回非零退出码；验收的 FAIL 和 INCONCLUSIVE 计入最终结果，退出码分别为 1 和 2，只有执行的检查通过才返回 0。
- Swift 包自检直接运行时显式从脚本路径确定默认目录，避免空 Path 参数导致自检无法执行。
- 负对照要求 NONE；新增缩短历史、真实生成至 512 token 上限、工具结果回传及最终回答、Messages/Responses、图片输入。保留接口边界：多工具 required/any 按 Auto；strict 不保证 Schema。
- Windows PowerShell 5.1 模拟回归 48 项通过；RTX 3060 12GB 上两套引擎小窗口验收均 exit 0（含 12,326 token 负对照、12,244 token 藏针）。没有重新测满 256K/32K、性能或困惑度。
- 引擎 C++ 与二进制不变，本轮没有新的容量或速度收益；保留已有本地鉴权配置方式。

English: launcher recovery now uses readiness, client limits follow retries, and verification distinguishes pass/fail/inconclusive. Real tool-result, output-cap, API and vision checks passed on an RTX 3060 12GB. The model-specific 0.1.1 source releases include these fixes; existing 0.1.0 cloud links remain unchanged.

## 0.1.0 - 2026-10-04

**源码与懒人包均已发布。** 本次给 Swift 1.5 Q2S 和 Bonsai2 12G 统一建立软件版本号 0.1.0；8G 方案继续使用此前的发布包。

- Swift 1.5 Q2S 与 Bonsai2 12G 分别发布了[独立源码 Release](https://github.com/5258MF/ninfer-rtx3060-27b/releases/tag/swift15-v0.1.0) / [Bonsai2 Release](https://github.com/5258MF/ninfer-rtx3060-27b/releases/tag/bonsai2-12g-v0.1.0)，各自的下载附件只含对应源码目录。主仓库整合快照标签 [`v0.1.0`](https://github.com/5258MF/ninfer-rtx3060-27b/tree/v0.1.0) 继续保留；源码基线为 [209c75b](https://github.com/5258MF/ninfer-rtx3060-27b/commit/209c75b206e2c12e18fce340526e3261979ec549)。
- 仓库根目录、`swift15/`、当时名为 `bonsai2-12g/`（现为 `bonsai2/`）各自提供 `VERSION`，值为 `0.1.0`。后续发布使用新版本文件和 Git 标签保留快照，`main` 继续开发。
- 两款包的百度、夸克新链接和提取码已更新到[下载表](README.md#下载)。2026-10-04，用户确认此前 Swift 夸克分享链接错误；现已按最新提供的版本化文件名、链接和提取码更正，旧地址不再作为下载入口。
- 包含下面记录的显存预算、长对话/看图缓存、三项智能体兼容修改和 Bonsai2 完整源码公开；已知接口限制继续适用。

| 方案 | 0.1.0 文件名 | 引擎 MD5 |
|---|---|---|
| Swift 1.5 Q2S | `ninfer-3060-swift15-q2s-mtp-oneclick-0.1.0.zip` | `DFC837D004E99BB297FFA982BAABCB5B` |
| Bonsai2 12G | `ninfer-3060-12g-bonsai2-oneclick-0.1.0.zip` | `6449090F250188CA9C1EE72A526CF74A` |

本次更新发布标识和下载入口，不重新测量性能或运行推理；原有测试范围和接口限制保留。下面未编号的记录说明 0.1.0 发布前各阶段的改动，旧文件名与哈希属于对应历史构建。

## 2026-10-04：三项智能体兼容修改与 Bonsai2 12G 源码公开

**状态：已包含在 0.1.0 的源码与两款网盘包中。** 本节记录该版本发布前的源码同步与接口边界。

| 方案 | 完整源码 | 启动器 | 编译入口 |
|---|---|---|---|
| Swift 1.5 Q2S | [`swift15/engine/`](swift15/engine/) | [`swift15/oneclick-iq2s/`](swift15/oneclick-iq2s/) | [`build-sm86.bat`](swift15/scripts/build-sm86.bat) |
| Bonsai2 8G / 12G 共用引擎 | [`bonsai2/engine/`](bonsai2/engine/) | [`oneclick-8g/`](bonsai2/oneclick-8g/) / [`oneclick-12g/`](bonsai2/oneclick-12g/) | [`build-sm86.bat`](bonsai2/scripts/build-sm86.bat) |

Bonsai2 首次公开的是完整 CMake 工程、实现、应用、测试、工具和第三方源码，包含此前 sm_86、SM 自适应、KVMem、rk8v4、显存预算、连接保活与长生成/多轮缓存修复。保留上游 LICENSE、NOTICE 和第三方许可；本机虚拟环境、评测运行记录、下载的语料、历史回退副本、私人配置、模型、DLL 与 EXE 不进入源码提交。

三项修改按当前源码同步：

1. **压缩、编辑历史与分叉的 KVMem 缓存保护。** 仅复用上一轮末尾的直接续写；中途截断、分叉和稳定前缀候选回退到 root 重新预填。预留额度不满足时跳过候选，避免继续沿用无效候选。回退会增加该请求的读入时间。
2. **输出与思考预算限制。** Chat、Messages、Responses 将过大的客户端输出请求按服务端 `default_max_tokens` 封顶，核心进一步受剩余上下文容量限制。通过原始请求校验后，思考预算也按实际输出与容量裁剪。Bonsai2 补齐 `reasoning_effort` 别名映射：none 关闭思考、minimal/low 映射 Low、medium 映射 Medium、high/xhigh 映射 XHigh。
3. **工具声明参数适配。** OpenAI 的 `required` / Anthropic 的 `any` 在单工具时指定该工具，多工具按 Auto 处理。接收相关并行控制声明；未强制的内置工具声明按现有实现处理。指定工具时关闭本轮思考，避免该组合直接报错。

当前实现仍有以下边界，不能据此声明所有智能体接口都完全兼容：

- 多工具 `required/any` 降级为 Auto，不保证每轮一定调用工具。
- `strict:true` 被接受的入口将其作为声明处理，不提供严格 JSON Schema 约束；**Swift Responses 的函数工具仍明确拒绝 `strict:true`**。其它工具字段也仍受各入口的校验规则限制。
- 未强制的内置工具声明被忽略，不代表引擎会执行网页搜索、MCP 等宿主工具。
- Anthropic 思考预算仍需满足原始请求的基本校验，例如正数/最小值和 `budget_tokens < max_tokens`；封顶逻辑发生在这些校验之后，不能消除所有 400。
- `<think>` 块内部工具识别保持原实现，本次没有合入用户未要求的第 4 项。

本机对应引擎：Swift MD5 `DFC837D004E99BB297FFA982BAABCB5B`、SHA256 `e83e4c18b6d92b37d3986c7c4c83b7d87b943776c70393ebcf8fc81429117a87`；Bonsai2 12G MD5 `6449090F250188CA9C1EE72A526CF74A`、SHA256 `3f137098e3ac1dc9c4695de75cbc44170db38cdf6594fb98573d80a9e1e73e4a`。源树与发布包分别标识，自行编译不会得到同一二进制哈希。

本次同步核对源文件哈希、两套现有 CUDA 13.3 / sm_86 构建、PowerShell 语法、文档链接、校验清单与新增公开内容；实机功能结果来自已有交接记录，本次未重新跑 GPU 推理、完整单测或性能测试。此前完整旧 KV 单测的三项断言失败未被算作通过。

## 2026-10-04：Swift 1.5 Q2S 源码与启动器同步

**状态：已包含在 0.1.0 的源码与 Swift 网盘包中。** 模型文件和转换方式不变，已有模型可以继续使用。

- 引擎的显存预算在 Windows 上结合 CUDA 和 NVML 的空闲显存，NVML 可用时取较小值；支持 `NINFER_FREE_VRAM_MIB` 限制，并输出预算日志。
- KVMem 内容相关性打分覆盖单段查询和提示窗口边界，统一输出 `kvmem_score: SELECT`。
- Windows 增加 TCP Keep-Alive（10 秒 / 3 秒），HTTP 连接保活 120 秒、读写超时 300 秒。
- 启动器缩窗重试时，必要时联动重算自动输出；启动前检查伴随 DLL。
- 启动器开放正文采样、自动恢复和自适应 MTP 开关；默认开启 `--post-thinking` 和 `--recover-invariant-failures`，自适应 MTP 默认关闭。
- 同步包内完整性检查与引擎验收脚本。公开验收脚本移除本机模型路径，使用包内相对路径或显式 `-ModelPath`；版本说明对齐本机最新引擎。
- 多轮看图时先完成 KV 预留与事务提交，再启动视觉编码借页；同时同步 KVMem 稀疏预留额度及预填/解码阶段处理，修复视觉借页抢占本轮 KV 导致的 `context cache exhausted`。

本机对应引擎 MD5：`931897FB80D42F9E01A51E0933F948F4`；SHA256：`ceece6ab31c72e5b0f0d89e68ad55d96b9b1691aae2922817ba8551f58da2c24`。EXE、DLL 和模型不作为源码提交。

本次同步核对源码哈希并通过现有 sm_86 / CUDA 13.3 构建检查；另外检查启动器及验收脚本语法、文档链接和公开内容。本次没有重新运行 GPU 推理或完整单测，原有性能和质量结果保持此前的测量口径。

## 2026-10-03：Swift 1.5 首次显卡校准修复

**状态：修复已随当时的完整包发布，也包含在 0.1.0 中。** 下表标识 2026-10-03 的历史构建；当前下载入口见[项目首页](README.md#下载)。

修复精简版自动校准仍调用未编译 E8 内核的错误。默认 rk4v4 也可能在没有校准记录的显卡首次启动时触发。引擎修复提交：[028908d](https://github.com/5258MF/ninfer-rtx3060-27b/commit/028908d00be94316505e62d76200b5231ef3fc7f)。

| 构建识别信息 | 值 |
|---|---|
| 文件名 | `ninfer-3060-swift15-q2s-mtp-oneclick.zip` |
| ZIP 大小 | 1,083,293,337 字节，约 1.01 GiB |
| ZIP SHA256 | `2882f5c52feeb9dcc618908f5f816ac14717f5f9715bcd104c22c5914154fc4f` |
| 引擎 MD5 | `F2CAAB583EA48ADD1566B6BA163DC44E` |
| 引擎 SHA256 | `9337b2668c72b940f5cbaf74c3c49dd552fc6405a867939f00594434ff904e32` |

验证：旧引擎强制首次校准复现原始 FATAL；修复版首次校准、缓存再次启动、接口回答通过。新 ZIP 经 CRC 校验，重新解压后默认启动器能正常回答，三个 VC DLL 实际从包目录加载。包内附版本信息，不包含模型和私人配置。

此引擎同时包含下面的容量曲线修复。旧容量启动器小补丁不包含首次校准修复；旧包处理方法见故障排查。

## 2026-10-03：Swift 1.5 容量曲线修复

**状态：完整源码已合入，此修复已包含在当前 Swift 1.5 网盘包中。**

- 修复实际读入分段经 SM 对齐后变大，而自动 KV 容量仍按请求值估算的问题。
- 使用规划器容量曲线的最低页数，保留 SM 自适应和分段对齐。
- 此前容量修复版引擎 MD5 为 `5998263BECD56D84A0A2954D7BE37CBF`。
- 另提供约 20 KB 的旧包启动器补丁，只针对这条容量错误重试；模型不需重新下载。
- RTX 3060 12GB 上完成复现和默认 200K 启动、接口生成验证；报错的 Laptop 现场未在本机直接验证。

## 2026-10-03：Bonsai2 12G 第十轮本机构建

**状态：本轮改动随后已包含在 0.1.0 中，12G 完整源码也已公开。** 下列文件名、大小和指纹记录第十轮初版构建。

- 实际 SM 数量识别，按空闲显存分配 KVMem 窗口和输出。
- 自动输出 8K–32K、检索窗口优先 36K；手动 64K 输出按实际数值计算。
- 容量报告触发有限缩窗重试；统一分段 512、512 行查询、换行边界截取、保留区域排除和批量 gather。
- 修正公共启动器的 PowerShell 上限表变量冲突，向导支持 auto / 0。
- 新包沿用 `ninfer-3060-12g-oneclick.zip`，333,426,250 字节，约 318 MiB；引擎 MD5 为 `B7A237671266B3BBD8B7FC255AC1A58A`。
- 长文、困惑度和启动验证范围见[第十轮测试结果](README-12GB.md#第十轮补充验证)。没有实测出明显提速、回答质量或上下文容量提升；未把完整旧 KV 单测列为通过。

## 2026-10-01：Bonsai2 12G 容量提升

**状态：当时的 12G 网盘已发布，此容量优化保留在 0.1.0 中。**

词嵌入和视觉权重放内存，合计节省约 600 MiB 显存。普通上下文从 64K 提高到 88K，rk8v4 从 88K 提高到 112K；KVMem 两种模式的显存容量上限分别为 76K / 100K，默认输出 32K。

这是此前的容量提升，不能归到后续第十轮更新。历史优化、测量和限制见 [12G README 的技术实现](README-12GB.md#技术实现)与[测试结果](README-12GB.md#测试结果与已知限制)。

## 早期修复汇总

以下为此前已合入的修复，不表示本次文档整理又修改了引擎。

### Swift 1.5

- 长对话里发一条很长的新消息，引擎会崩溃：已修；
- 超窗复用时的几处校验过严、接续点丢失：已修。
- 部分显卡的 SM 数导致实际读入分段变大，KV 容量预算没有同步，引擎启动时报容量曲线越界：修复已合入最新源码。

### Bonsai2 12G / 8G

- **多轮对话必崩（原版引擎的已知缺陷）**：续写时有一个长度校验写错了，第二轮就报错，而且会让整个引擎不可用。现在修好了。
- 做 KVMem 过程中又修了几个缓存管理的问题，比如"重新生成"或改最后一条消息时引擎挂掉、闲置缓存清出后再复用报错、接近上限的长提示报错。现在四种模式都经过了长文、多轮、工具循环、看图的回归测试。
- 修正早期评测工具里的内存分配错误。

## 启动错误与处理

下面两项引擎修复针对 Swift 1.5 Q2S 包。版本识别信息见本页上方；普通使用、配置、测试和编译说明保留在各版本 README 中。

### KV 页数超出容量曲线

报错：`Main KV page count is outside the target capacity curve`。

旧引擎按请求的读入分段估算容量，而实际分段会按 SM 数对齐，例如 512 变成 640，导致页数低于规划器下限。[当前源码](swift15/engine/src/runtime/engine/model_instance.cpp)已使用规划器容量曲线的最低页数。

更新引擎后继续使用启动器生成的参数。KVMem 的总上下文在内存，显存只保留窗口；`--max-context 204800` 大于 `--kvmem-window-pages 1152` 对应的窗口是正常配置，不需要因此删除窗口参数。

旧包也可以使用[容量启动器补丁](swift15/hotfix/swift15-hotfix-kv-capacity.zip)：关闭引擎，解压到原包目录覆盖 `launcher/launch.ps1`。它只在遇到这条错误时关闭分段对齐并重试一次，不能修复下面的首次校准问题。

### 首次显卡校准失败

报错：`small_t_i8: this NINFER_SLIM_3060 build supports only int8 / rk8v4 / rk4v4 KV caches`。

精简构建没有编译 E8 内核，但旧自动校准仍会测试它们。显卡没有匹配校准记录时，即使用默认 rk4v4，也可能在加载模型前失败。[当前源码](swift15/engine/src/calibration/device_calibration.cu)已让精简版只校准实际编译的 int8 / rk8v4 / rk4v4。

优先使用包含修复的新引擎，识别信息见更新记录。旧包可先在包目录的 CMD 窗口运行：

```bat
set "L8084_EXTRA_ARGS=--device-profile off"
启动.bat last
```

这会跳过路线校准，SM 数量仍会识别。更新引擎后，在新 CMD 窗口正常运行 `启动.bat` 即可。首次正常校准可能需要一段时间，请等待 `listening`。

### VC 运行库或其他 DLL 加载错误

若 Windows 提示 DLL 找不到或入口点不存在，先确认完整解压，保持 EXE 与所有 DLL 在 `engine` 目录中。所需 VC 文件包括 `msvcp140.dll`、`vcruntime140.dll`、`vcruntime140_1.dll`。

需要修复系统运行库时，使用[微软官方 x64 VC 运行库](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)。新版包已经检查静态及延迟导入依赖，并实际验证这三个 VC DLL 从包目录加载；[微软本地部署说明](https://learn.microsoft.com/en-us/cpp/windows/determining-which-dlls-to-redistribute)允许这种同目录部署方式。

DLL 加载错误与上面的 E8 校准逻辑错误分别处理。安装运行库不会补回精简构建没有编译的 E8 内核。
