"""Data model for the topology-aware GPU placement planner.

Link levels, from fastest to slowest:

- ``nvlink``: GPUs inside one node connected by an NVLink clique.
- ``pcie``:   GPUs inside one node that only share PCIe.
- ``leaf``:   GPUs on different nodes behind the same leaf switch (rail).
- ``spine``:  GPUs on nodes that only meet at the spine layer.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

LINK_LEVELS = ("nvlink", "pcie", "leaf", "spine")


def link_rank(level):
    """Lower rank means a faster link."""
    return LINK_LEVELS.index(level)


def weakest(levels):
    """Return the slowest link level from an iterable of levels."""
    levels = list(levels)
    if not levels:
        return None
    return max(levels, key=link_rank)


def link_satisfies(available, required):
    """True if link level ``available`` is at least as fast as ``required``."""
    return link_rank(available) <= link_rank(required)


@dataclass
class Node:
    """A compute node and its GPU topology.

    ``nvlink_cliques`` lists sets of GPU indices fully connected by NVLink;
    GPUs not sharing a clique talk over PCIe. ``numa_of_gpu`` maps a GPU
    index to its NUMA node id. ``leaf``/``spine`` describe the switch
    hierarchy. ``fault_domain`` defaults to the node name (each node is
    its own failure domain).
    """

    name: str
    gpu_model: str
    gpu_count: int
    gpu_memory_gb: float
    free_gpus: Optional[int] = None
    nvlink_cliques: List[Set[int]] = field(default_factory=list)
    numa_of_gpu: Dict[int, int] = field(default_factory=dict)
    leaf: str = "leaf-0"
    spine: str = "spine-0"
    fault_domain: str = ""

    def __post_init__(self):
        if self.free_gpus is None:
            self.free_gpus = self.gpu_count
        if not self.fault_domain:
            self.fault_domain = self.name
        if not self.numa_of_gpu:
            self.numa_of_gpu = {idx: 0 for idx in range(self.gpu_count)}

    def cross_link(self, other):
        """Link level of traffic between this node and ``other``."""
        if self.leaf == other.leaf:
            return "leaf"
        if self.spine == other.spine:
            return "spine"
        return None


@dataclass
class JobRequest:
    """Placement request for one training job.

    ``min_link`` is the slowest link level the collective communication
    shape tolerates: ``nvlink`` forces a single NVLink clique, ``pcie``
    forces a single node, ``leaf`` allows spreading across one leaf
    switch, ``spine`` allows the whole pod. When ``fault_domain_aware``
    is set, every replica must sit inside exactly one fault domain and
    no two replicas may share a domain.
    """

    name: str
    replicas: int
    gpus_per_replica: int
    gpu_model: Optional[str] = None
    min_gpu_memory_gb: float = 0.0
    min_link: str = "spine"
    numa_local: bool = False
    fault_domain_aware: bool = False

    def __post_init__(self):
        if self.min_link not in LINK_LEVELS:
            raise ValueError("unknown min_link: %r" % self.min_link)
        if self.replicas < 1 or self.gpus_per_replica < 1:
            raise ValueError("replicas and gpus_per_replica must be >= 1")
