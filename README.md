# Chem-Mamba

**用选择性状态空间模型（Mamba/SSM）为机器学习原子间势（MLIP）提供 O(N) 的非局域长程模块。**

## 核心想法（一句话）

主流 MLIP 依赖 locality 假设，长程/非局域效应处理不好。最强的长程方法 Latent Ewald
Summation（LES）用 **局域** latent charge，结构上无法表达 **全局电荷再分配 / 电荷转移**；
历史上只有 4G-HDNNP 用 O(N³) 电荷均衡解决了这个问题。

> **Chem-Mamba：选择性 SSM（Mamba）在一次线性扫描中"看过"整个体系，提供 O(N) 的非局域
> 全局上下文——是"局域 GNN"与"O(N³) 全局 QEq"之间缺失的中间态；配合物理正确的
> latent-Ewald 尾部保证外推。**

完整论证、架构、实验方案与风险分析见 **[`docs/PROPOSAL.md`](docs/PROPOSAL.md)**。

> 注:`docs/`(研究笔记、完整结果台账、论文草稿)与 `data/`(Ko et al. 2021 四基准,
> 从 [Materials Cloud bk6kw-4b895](https://archive.materialscloud.org/record/2020.92) 下载)
> 在论文发表前暂不随仓库公开,故仓库内相关链接暂不可用。

## 概念验证（`poc/`）

在可控的电荷均衡（QEq）非局域任务上，隔离检验唯一核心假设：*线性标度的 SSM 能否捕捉纯
局域模型结构上无法捕捉的非局域电荷/能量分配？*

| 文件 | 作用 |
|---|---|
| `poc/qeq_data.py` | QEq 真值数据生成 + 判别性测试集（冻结局域邻域、只变远场） |
| `poc/models.py` | 三个共享特征化的模型：局域 DeepSet / 双向选择性 SSM(Mamba S6) / 全注意力 |
| `poc/run_poc.py` | 训练 + 评估 + 判别性测试，输出对比表 |
| `poc/scaling.py` | 前向时间/显存 vs N 的标度对比（SSM 线性 vs 全对注意力二次） |

运行：

```bash
cd poc
PYTHONPATH=. python run_poc.py      # 主实验（约 10 分钟，含朴素扫描的 SSM 训练）
PYTHONPATH=. python scaling.py      # 标度实验（约 2 分钟）
```

## 关键结果

**判别性测试**（标记位点局域邻域冻结、仅远场变化，真值 `std(q_marked)=0.096`）：

| 模型 | 复杂度 | 电荷 MAE | 能量 MAE | 标记位点相关性 |
|---|---|---|---|---|
| local（=LES 局域电荷） | O(N) | 0.036 | 0.057 | **−0.00**（结构性失效） |
| **ssm（本文）** | O(N) | **0.020** | **0.017** | **0.896** |
| transformer（全注意力） | O(N²) | 0.092 | 0.308 | 0.466 |

- 局域模型相关性 = 0：其局域输入恒定 ⇒ 输出恒定，**信息论意义上无法** 追踪非局域变化。
- SSM 相关性 = 0.896：线性标度扫描成功捕捉全局电荷再分配。

**标度**：SSM 时间/显存随 N 线性；显式全对注意力显存二次，大 N 撞显存墙——16384 原子时
注意力用 8.6 GB（约 32–64k 即 OOM），SSM 仅 654 MB。

（完整表格与解读见 `docs/PROPOSAL.md` §4。）

## 状态

- [x] **M0 概念验证**：机制成立（`poc/`）
- [x] **M1 接真实 3D 等变 backbone 端到端**（`chem_mamba/`，见 [`docs/M1_RESULTS.md`](docs/M1_RESULTS.md)）
  - ✓ 对称性正确：能量对旋转/平移/置换不变、力等变，均达机器精度（~1e-7），力经 float64 有限差分验证（6.7e-09）
  - ✓ 端到端能量+力可训、不倒退短程（SSM ≥ local）
  - ⚠️ 诚实结论：容量匹配消融（同参数、只关/开跨原子 reach）显示 global ≈ isolated（ΔE +0%）——**SSM 增益全来自容量、非局域 reach 零收益**，因为本合成设置的电荷**局域可预测**（即"long-range 其实没那么难"的重现）。
  - 📌 **关键教训**：非局域收益是**体系**的性质、不是维度的性质；必须在**真正非局域**的体系（M3 的 4G 电荷转移基准）上展示。capacity-independent 铁证是 1D 概念验证（专门构造的判别任务）。
- [x] **M3 决定性实验：4G-HDNNP 电荷转移基准**（真实 DFT 数据，Ko et al. 2021 四个体系，
  见 [`docs/M3_RESULTS.md`](docs/M3_RESULTS.md)）
  - ✓ **NaCl（Q 零信息体系）**：容量匹配后非局域 reach 净增益 **电荷 −37%、能量 −16%、力 −20%**；
    位移 Na 电荷追踪 corr **0.96 vs 0.49**（ssm vs 容量匹配对照）——首次在真实数据上排除容量
    因素测得非局域收益
  - ✓ **碳链**：local(=LES 式) 远端 H 电荷物种间距 −2.1 me vs 真值 +43.2 me（结构性失效实证）；
    ssm 电荷 5.23 me 达 4G-HDNNP 文献区间（4.8–5.0 me），用 O(N) 而非 O(N³)
  - ✓ **Ag₃ 对照**：Q-aware 变体打平（无虚假增益）、local-Q 灾难失效（1259 meV/atom）
  - 🔄 AuMgO 旗舰（Al 掺杂距 Au₂ ≥10.3 Å、Q 恒 0）运行中
- [ ] M2 latent-Ewald 尾部 + 周期性 + 外推实验（体相水/电解质，对比 LES）
- [ ] M4 色散基准 + 大体系标度 + 多种子 + 强 backbone + 写作

`chem_mamba/`：`layers.py`（Mamba S6/双向块）· `data3d.py`（3D 可微参考势）·
`model.py`（SchNet 风格 backbone + SSM 长程模块 + 力 autograd）· `train_m1.py`（训练+对称性门槛）。
