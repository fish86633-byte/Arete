# 会议论文实验与来源

## 当前仓库可以直接运行什么

安装后可独立运行 Arete 的单个完整规划时域、生产队列回放、独立核验和合成示例，不依赖旧项目的 Python 导入路径。

`paper_config` 提供参数，不会读取 GHS-POP、生成轨道、生成 Poisson 业务或重建冻结种子。正式论文结果来自下列实验档案。完全重现这些结果还需要其冻结输入、连续状态、驱动和协议；本次没有把这些大体积资产打进代码仓库。

## 原实验位置

下表路径均相对工作区 `/Users/fish/Desktop/波束/`；机器迁移后通过 `provenance/experiment_index.json` 重新定位原档案。

| 证据 | 原目录 / 文件 |
|---|---|
| 四档负载完整结果 | `experiments/dams_u8_load_030/results_load_030_resume_031_tail13/report/` |
| 负载输入与登记 | `experiments/dams_u8_load_030/results_load_030/` |
| rho=0.8 三段连续状态来源 | 同项目 `results_load_030`、`results_load_030_resume_031`、`results_load_030_resume_031_tail13` |
| 星座规模 200 周期结果 | `experiments/dams_constellation_recovery_035/results_recovery_035/report/` |
| 星座前缀 / 核验加速 / 续跑来源 | `experiments/dams_constellation_200_033`、`dams_audit_acceleration_034`、`dams_constellation_recovery_035` |
| GA1 固定种子结果与续跑 | `experiments/ga_bhtp_continuous_036/results_ga_036_ga1/` |
| NX 六个配对输入 | `experiments/pcqr_bh_r3_local_repair/results_supplement_013/report/nx_paired.csv` |
| NX 原始实现和驱动 | `experiments/pcqr_bh_r3_local_repair/supplement_013/` |
| DPG / BG 实现 | `experiments/dams_u8_load_030/dpg_control_016/`、`backlog_control_020/` |
| RR / TPG 及兼容基线原语 | `src/arete/_frozen/pcqr/baselines.py`，正式驱动以原协议为准 |
| GA 算法及冻结登记 | `experiments/ga_bhtp_continuous_036/ga036/`、`audit/registration.json` |

这里没有把六算法实验驱动改写成新的统一驱动，避免在整理过程中改变预算、初始化、种子或队列接续。GA/BG/DPG 等正式比较方法仍从各自档案定位，不宣称本仓库已经封装全部基线。

## 轻量结果副本

`paper_results/` 保存现有负载、星座、GA1 和 NX 的小型汇总，以及原论文绘图数据 JSON。文件逐字节复制，数值和历史方法名保持原样；DAMS/M1-E 对应 Arete。绘图 JSON 中可能保留 DAMS 键名，其显示名称由原绘图源决定。

这些副本用于核对论文和寻找来源，不能替代逐周期证据。`provenance/experiment_index.json` 登记原路径、复制后路径和 SHA256。

## 统计口径

- 主负载与星座结果：每方法、每流 200 周期，共 20 s；前 2 s 预热，后 18 s 评价。40 周期结果是历史短窗结果，不能替代主结果。
- 两条业务流等权平均。GA 主结果统一固定 GA1 基种子 351001，旧 GA2/GA3 不混入主均值。
- RT 过期率按已经成熟的期限到达集合统计；时延仅计算成功交付的 RT 包。两个流的 P95 平均不是合并数据包 P95。
- 小区满足率按原评价窗口及到达集合定义，不能用另一套交付/到达截断公式重算后冒充原值。
- Table II 的 Delta F、Delta R、Delta W_RT 是每个 k 下两个配对输入的均值；NX 负净边际加入次数按两个输入求和，最后一列不是 Arete−NX 的差值。
- 连续周期不是独立重复；两流结果不支持跨种子稳健性或独立周期显著性结论。200 周期也不自动表示队列收敛。

本次整理没有新增正式仿真、改写历史停止状态、重新核验旧前缀或改动论文。合成示例的运行时间不用于比较论文算法速度。
