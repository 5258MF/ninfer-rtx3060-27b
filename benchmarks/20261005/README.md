# 2026-10-05 / 06 评分与性能记录

**简体中文 | [English](README.en.md)**

主展示表见项目中文/英文 README。本目录保存可核对的评分报告及去掉提示词、生成文本后的测速数据。K = 1024 token。

- `ppl/report-01.json`：IQ3XXS MTP Q4；`report-02.json`：同一主模型的原始 MTP Q6；`report-03.json`：Bonsai2 历史 quick 评分。
- `ppl/report-04.json` 至 `report-19.json`：早期 custom-tiny / custom-wiki40k 语料的流水线检查和不同 context/stride 测量，不能与 quick 质量结果混算。
- `ppl/catalog.json`：19 份原始报告的模型文件名、语料、执行设置、评分规模和 SHA256。文件内容与归档原报告一致；历史重复测量也保留。
- `performance/decode-samples.json`：10 个配置的文本/代码计时，保留预热标记、原始 usage/timings、生成文本 hash 与源报告 hash；未公开提示词和生成内容。
- `performance/prefill-audit.json`：10 个配置 × 2K/8K/16K；每档预热 1 次、正式 3 次，共 90 正式样本。引擎 prompt_n 与 usage 相符，缓存为零；最多生成 8 token。吞吐和 TTFT 分开列出。

IQ3 PPL：quick 四个流、4096 context / 2048 stride、rk4v4、GDN FP16、embedding-host、评分预填充分块与 score tile 均为 1024。独立评分入口增加 embedding-host 选项以容纳模型，概率计算及语料未变。每文件评分 261167 token、124 窗口；总体 PPL 4.495637819904494，Q6/Q4 主模型评分在报告精度下相同。CausalScoring 不运行 MTP 草稿或视觉，不能用此指标证明接受率、草稿质量或多模态能力不变。

速度使用 RTX 3060 12GB、Windows、CPU 视觉、完整输出头、MTP draft3、聊天 prefill chunk512。文本/代码为短输入，输出上限256，每组预热1次再测3次，表中为中位数。不同模型驻留容量不同；同一模型/KV的 Graph 对照使用相同容量。受温度、桌面和顺序运行影响，小差异不构成普遍提速结论。达到256 token长度上限不代表生成的代码完整或正确。

这些性能记录使用当时的测试窗口：IQ3 37K、Bonsai8 48K；当前启动配置已改为 IQ3 33K、Bonsai8 驻留上限36K。不能把测速窗口当成当前默认或重新通过的容量验收。IQ3 prefill 首次尝试因剩余223.68MiB触发旧300MiB保护，重试完成；原异常记录保留在 audit 中。2K–16K prefill 均未超过驻留窗口，不代表超窗KV换入速度。没有以短请求推断32K/64K连续长输出稳定性。

19份PPL原报告全部保留，但只有设置匹配的报告能形成可归因A/B。旧IQ2S PPL表本次没有找到对应原报告，保留为历史参考。其余原始实验和会话保留在用户本地归档；不把测试DSH入口、日志或会话作为通用启动包发布。

English: Original PPL reports and prompt-free performance metrics. IQ3 main-model PPL is 4.495637819904494, unchanged between MTP Q6/Q4 artifacts under the recorded quick scoring settings. Decode medians and zero-cache 2K/8K/16K prefill results are measured in different test windows from current defaults. Early custom-corpus reports are historical pipeline/sweep data. These are not long-output, draft-quality or broad stability certifications.
