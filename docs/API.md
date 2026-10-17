# 输入与输出

## 一个规划时域

```python
import arete

config = arete.paper_config(rho=0.4)
state = arete.State.empty(M=400, ttl=config["ttl"])

# 由调用方提供真实输入，不在 plan 内随机生成。
result = arete.plan(
    state, arrivals, satellite_positions, cells, satellite_ids,
    config=config, cycle=0,
)
if result["run_status"] != "OK" or not result["output_published"]:
    raise RuntimeError(result.get("error", result["stop"]))

audit = arete.verify_plan(
    result["plan"], state, arrivals, result["channel"],
    start_slot=0, expected=result["trace"],
)
next_state = audit.final_state
```

`State.empty` 的 M 为当前区域小区数。下一次规划需要生产模型的 State，可使用
`arete.State(next_state.rt, next_state.nrt_batches)`；有完整生产 trace 时也可直接使用
`result["trace"].end_state`。零预算安全回退没有生产 trace，此时应先实际执行或独立回放方案，再推进队列。

| 输入 | 形状 / 含义 |
|---|---|
| `satellite_positions` | 浮点数组 `[H,S,3]`，ECEF 坐标，单位 m |
| `cells` | 浮点数组 `[M,3]`，地面小区中心 ECEF 坐标，单位 m |
| `satellite_ids` | 整数数组 `[S]`，唯一全局卫星编号，用于确定性排序 |
| `arrivals` | 非负整数数组 `[H,M,2]`，最后一维依次为 RT、NRT 包数 |
| `state.rt` | 非负整数数组 `[M,D]`，按年龄 0 至 D−1 索引的 RT 包数 |
| `state.nrt_batches` | 每小区的 `(绝对出生槽, 包数)` FIFO 批次，不能仅保留总数 |
| `cycle` | 当前规划周期，首个绝对槽为 `cycle*H` |

输入队列位于第一槽到达加入之前。到达先于服务，RT 按年龄优先服务，NRT 保留 FIFO，随后按原规则过期和老化。多个算法、业务流的队列必须各自独立。

S 是本次传入的卫星集合长度，不是参数中的整个星座卫星总数。可传入完整几何，也可按原实验协议传入当前时域的卫星子集；服务资格仍由每槽可见性决定。仅把 `config["planes"]` 改成 12 或 36 不会改变轨道几何。

## 输出

`plan` 返回原冻结实现的审计字典，另加 `algorithm` 和 `channel`：

- `plan`：布尔数组 `[H,S,M]`；每星每槽最多 K 束，每小区每槽最多一颗服务星。
- `trace`：完成的完整时域回放，包含精确 Fraction 目标 F、交付 bits、W_RT、H+1 个队列状态；早期安全回退时可为 None。
- `initial_plan`：该次调用自己计算的 TPG 参考方案，初始化未完成时可为 None。
- `accepted`：构造方案是否通过完整时域接受条件。
- `run_status`、`stop`、`output_published`：计算与输出状态；TIME_LIMIT 不等于正确性失败。
- `reliable_plan_source`、`fallback_used`：实际返回的完整方案来源。
- `costs`：规划墙钟、CPU 和各阶段费用；部分子阶段为嵌套计时，不能重复相加。
- `proposal.construction`：候选池、逐步选边、净边际收益和筛选证据。

保留 `Fraction`，不要在验收或选边前转成浮点数。`audit.summary()` 可用于输出 JSON 数值摘要。

## 预算与并发

公开入口在构造冷 Channel 前开始计时。输入文件读取、包导入和独立核验在规划计时之外；总实验费用仍需另行计量。2 s 是合作式软预算，原子操作可能超额。

公开入口在单个进程内串行执行，保护原实现的临时构造器绑定。需要并发时使用独立进程和各自的状态、Channel；不要绕过入口并发调用 `_frozen` 模块。
