# training-gpu-topology-aware-placement-planner
按 GPU 拓扑、链路和故障域规划训练任务放置

## 概述

集群调度器只按 GPU 数量选节点时，同一训练任务的 GPU 可能跨 NUMA、
跨交换机轨道分配，集合通信（all-reduce 等）被最慢的一条链路拖垮。
本规划器在选节点时同时考虑链路等级和故障域，而不是只数 GPU 总数。

**重要：本规划器只计算放置方案，不启动任何任务。** 它输出候选节点、
预计链路瓶颈和无法满足的原因，由外部调度器决定是否真正下发任务。

## 链路等级

从快到慢：`nvlink`（节点内 NVLink 组）→ `pcie`（节点内 PCIe）→
`leaf`（跨节点、同一叶子交换机/轨道）→ `spine`（跨叶子、经骨干层）。
任务的 `min_link` 表示集合通信能容忍的最慢链路：选 `pcie` 则任务
必须落在单节点内，选 `leaf` 允许在同一叶子交换机下跨节点。

## 输入

- 节点：GPU 型号、GPU 数量与空闲数量、单卡显存、NVLink 组
  （`nvlink_cliques`）、GPU 所属 NUMA（`numa_of_gpu`）、交换机层级
  （`leaf`/`spine`）、故障域（`fault_domain`）。
- 任务：副本数（`replicas`）、每副本 GPU 数、GPU 型号与显存要求、
  `min_link`、是否 NUMA 对齐（`numa_local`）、是否故障域感知
  （`fault_domain_aware`，开启后每个副本独占一个故障域且整个副本
  不跨故障域）。

## 输出

`plan()` 返回 `PlacementResult`：

- `assignments`：每个副本的候选节点与 GPU 编号、该副本的链路瓶颈、
  所在故障域。
- `bottleneck`：整个任务预计的链路瓶颈（所有副本中最慢的一级）。
- `fragments`：部分空闲节点被使用后剩余的 GPU 碎片（节点名 → 剩余
  空闲 GPU 数）。
- `reasons`：无法满足时的原因（GPU 总数不足、NVLink 组太小、故障域
  数量不够、链路等级不允许跨节点等）。

## 示例

```python
from gpu_planner import JobRequest, Node, plan

nodes = [
    Node(name="n1", gpu_model="A100", gpu_count=8, gpu_memory_gb=80,
         nvlink_cliques=[{0, 1, 2, 3}, {4, 5, 6, 7}], leaf="leaf-1"),
    Node(name="n2", gpu_model="A100", gpu_count=8, gpu_memory_gb=80,
         nvlink_cliques=[{0, 1, 2, 3}, {4, 5, 6, 7}], leaf="leaf-1"),
]
job = JobRequest(name="train", replicas=2, gpus_per_replica=4,
                 gpu_model="A100", min_link="nvlink",
                 fault_domain_aware=True)
result = plan(job, nodes)
print(result.success, result.bottleneck, result.fragments, result.reasons)
```

## 测试

```bash
python -m unittest
```

覆盖单节点 NVLink、跨节点 PCIe/叶子交换机、故障域限制和碎片资源。
代码兼容 Python 3.8.2。
