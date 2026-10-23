# NL/Sleep Reproduction Lab

这是 Nested Learning 与 Language Models Need Sleep 的非官方、局部实现与诊断研究。
不是官方发布，没有复现全部架构或论文基准指标，没有机构背书。

安装与命令以 [英文 README](README.md) 为准。安装依赖后，可离线运行 CPU 合成时钟、
状态隔离、reset/activate、路由反例与数值测试，并从附带 CSV 重算历史汇总。
测试中小型 receiver 的更新只说明机制，不代表历史 3B 训练重演。

四类证据严格区分：CPU_MECHANISM_DEMO（合成机制）；BUNDLED_RESULTS_REANALYSIS
（已有聚合数值重算）；USER_SUPPLIED_ASSETS_REQUIRED（需合法自备资产）；
HISTORICAL_ONLY（关键私有状态不再分发，无法仅凭本包精确复演）。hash 不等于资产已提供。

原生模型最初五组各四条轨迹的平均准确率均为 25%，充分习得未建立。
后续六条 Teacher 完整习得门槛均失败。局部恢复与新旧取舍并存；Teacher 未学会的新能力
不能称被保留。小型探索性 dev 被重复使用，共享 Teacher、轨迹与配方不构成独立样本。

Llama 路径只在最后 MLP 增加两级 CMS，原骨干冻结；base 指 CMS base。
它是 CMS-inspired 扩展，不等于原生 HOPE。CACHE 约 3.40 倍只适用于特定重复 bank；
公开入口默认 REF，无可转移验收时拒绝显式加速请求，未知 scope 不扩大采用范围。
本轮 Llama/GPU 为 NOT_RUN。真正 64 边界、同时到期及终端生命周期仍未验证。

[技术报告](reports/technical_report.md) 与 [结果说明](results/README.md) 保留失败和缺失。
这些观察针对本实现/配置，不证明原论文无效。代码、文档、结果、模型和数据分别处理许可；
新增代码、测试、示例和工具采用 MIT；新增文档、报告及结果汇编采用 CC BY 4.0。
项目署名为 **Zhirui Zhu（zhiruizz）**，原有版权 notice 和第三方署名保持不变。
许可仅覆盖许可人有权授予的权利，不延伸到模型权重、原始数据或其他第三方资产。
