# 代码导览

公开 API 使用 Arete 名称；冻结依赖保留原实验模块名。27 个模块及 8 个包入口是公开规划器和核验器的静态依赖集合，不代表启动 27 种算法。

| 功能 | 仓库内位置 |
|---|---|
| 公开算法入口、预算、U8 绑定 | `src/arete/planner.py` |
| 自费 TPG、完整回放、严格目标接受及安全回退 | `src/arete/_frozen/confirmation_010/marginal_wrapper.py` |
| 逐槽构造、确认和后缀处理 | `src/arete/_frozen/speed_validation_017/construct.py` |
| 需求感知候选池 | `src/arete/_frozen/exploration_009/marginal.py`，`fixed_pool` |
| 精确净边际评分 | `src/arete/_frozen/confirmation_010/marginal_fast.py` |
| 认证上界筛选 | `src/arete/_frozen/speed_validation_017/upper.py` |
| U8 RT 年龄批处理缓存 | `src/arete/_frozen/speed_validation_018/scorer.py` |
| U8 构造器适配 | `src/arete/_frozen/speed_validation_018/construct.py` |
| 几何、天线、功率、整数容量 | `src/arete/_frozen/pcqr/physics.py` |
| RT/NRT 状态、槽执行、完整时域目标 | `src/arete/_frozen/pcqr/engine.py` |
| 完整 NRT FIFO 存储 | `src/arete/_frozen/pcqr/fifo.py` |
| 独立标量物理与 FIFO 核验 | `src/arete/_frozen/pcqr/verify.py` |
| 预算、消息编码和状态哈希 | `src/arete/_frozen/pcqr/support.py` |

其他 `exploration_*`、`prototype`、`pcqr.search` 模块是旧包装器的兼容依赖。公开入口执行 M1-E + U8，不执行旧 PCQR 的价格或局部修复流程。

后续期刊算法可另建模块及入口，与该快照并存。若确实修改 `_frozen/`，源码检查会失败；应先明确新算法身份和参数，再生成新的来源记录，避免把新结果与会议版混在一起。
