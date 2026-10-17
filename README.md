# Arete

**Demand-Aware Marginal Scheduling for Multi-Satellite Beam Hopping**

这是 Arete 会议论文实现的独立代码仓库，也是后续期刊扩展的代码起点。
论文中的 Arete 对应历史实验代码的 **M1-E + U8（UB）**。本次整理只搬移依赖、封装入口和补齐说明，没有更改调度规则、物理模型、队列模型或实验结果。

## 安装和运行

需要 Python 3.10 或更新版本，以及 NumPy、SciPy。在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m arete demo
```

`demo` 使用 2 颗静止卫星和 6 个小区的确定性合成输入，运行一个 20 槽规划，并独立核验物理计算、RT 超时和 NRT FIFO。它用于检查安装与接口，**不是论文实验数据**。运行时不会启动后台实验。

查看论文参数：

```bash
python -m arete config
```

已有 NumPy、SciPy 时，也可免安装运行：

```bash
PYTHONPATH=src python examples/one_horizon.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

本机现有的可用解释器是 `/Users/fish/Desktop/波束/multisat_bh_v1/.venv/bin/python`。新仓库运行时不需要导入该旧项目的代码。

## 代码结构

```text
Arete/
├── src/arete/
│   ├── planner.py       # 统一入口：冷几何、预算及 M1-E + U8 绑定
│   ├── model.py         # State、Channel、完整时域 replay、独立核验
│   ├── config.py        # 论文参数和物理常数
│   ├── demo.py          # 小型合成示例
│   ├── data/            # 随 Python 包分发的参数
│   └── _frozen/         # 从实际实验实现提取的 35 个源码文件
├── configs/             # 便于查看的论文参数 JSON
├── examples/            # 最小运行示例
├── tests/               # 原实现对照、物理/FIFO 核验、超时与连续推进检查
├── tools/               # 源码哈希及搬移范围检查
├── docs/                # 接口和论文复现说明
├── paper_results/       # 已有结果的轻量副本，不包含原始轨迹
└── provenance/          # 来源、版本、哈希及原始登记配置
```

开发首先看 `src/arete/planner.py`。核心算法和模型位置见 [代码导览](docs/CODE_MAP.md)，输入输出格式见 [API 说明](docs/API.md)。

## 论文实现与参数

- 算法实现提交：`34d7c2c207404a083f2af062957c45c754e1b504`。
- U8 工程实现提交：`8a452b2542f3656fd4b39c6612c98419aa53bfdf`。
- H = G_exec = 20，槽长 5 ms，K = 4，RT TTL = 60 槽，alpha = 0.5。
- 完整规划软预算 2 s；Q_max = L_eval = R_max = 1。
- 默认参数对应 24×22 星座、400 小区、rho = 0.4。参数本身不生成轨道或到达样本。

历史 `frozen_r2.yaml` 中的 PCQR 默认值没有作为本仓库默认值。当前参数从正式负载实验的登记配置提取；删去驱动元数据，并按同一 C_ref 派生 rho = 0.4 的 R_total，详见 `provenance/source_manifest.json`。

`_frozen/` 保留历史模块名和结果字段名，以便逐项核对来源。公开入口返回额外字段 `algorithm="Arete"`；内部证据中的 DAMS/M1-E 名称没有被改写。

## 核验和实验数据

```bash
python tools/check_sources.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

在本机额外核对原源码：

```bash
python tools/check_sources.py --original-root ../experiments/dams_u8_load_030
```

原始实验目录、完整 FIFO、逐周期轨迹、Git 历史和输入银行仍保存在 `../experiments/`。本仓库只带轻量结果副本与来源索引，没有复制大体积证据，也没有重新运行正式实验。正式实验的对应关系与复现边界见 [论文复现说明](docs/PAPER_REPRODUCTION.md)。

当前是本地研究仓库；尚未指定开源许可证或发布到远程平台。
