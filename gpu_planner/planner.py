"""Greedy topology-aware GPU placement planner.

The planner searches for a node combination that satisfies both the
link level and the fault-domain constraints of every replica, instead
of only counting total GPUs. Partially free nodes may be used; the
leftover fragments are recorded in the result.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .model import JobRequest, Node, link_rank, link_satisfies, weakest


@dataclass
class ReplicaPlacement:
    replica: int
    # List of (node_name, [gpu indices]) assigned to this replica.
    allocations: List[Tuple[str, List[int]]]
    bottleneck: str
    fault_domain: str


@dataclass
class PlacementResult:
    job_name: str
    success: bool
    assignments: List[ReplicaPlacement] = field(default_factory=list)
    bottleneck: Optional[str] = None
    # node name -> free GPUs left over after placement (only nodes the
    # planner allocated from, i.e. the fragmentation report).
    fragments: Dict[str, int] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)


def _clique_of(node, gpu):
    for clique in node.nvlink_cliques:
        if gpu in clique:
            return frozenset(clique)
    return frozenset([gpu])


def _intra_node_link(node, gpus):
    """Weakest intra-node link level among the given GPUs on one node."""
    if len(gpus) <= 1:
        return "nvlink"
    cliques = {_clique_of(node, gpu) for gpu in gpus}
    return "nvlink" if len(cliques) == 1 and len(next(iter(cliques))) > 1 else "pcie"


def _select_gpus(node, free, count, min_link, numa_local):
    """Pick ``count`` GPUs from ``free`` on ``node`` or return None."""
    free = sorted(free)
    if len(free) < count:
        return None
    if link_rank(min_link) <= link_rank("nvlink"):
        # Must fit inside a single NVLink clique.
        for clique in node.nvlink_cliques:
            cand = [gpu for gpu in free if gpu in clique]
            if numa_local:
                cand = _best_numa_subset(node, cand, count)
            if cand is not None and len(cand) >= count:
                return cand[:count]
        return None
    if numa_local:
        cand = _best_numa_subset(node, free, count)
        if cand is None:
            return None
        return cand[:count]
    return free[:count]


def _best_numa_subset(node, gpus, count):
    """Return GPUs from the NUMA domain with the most candidates."""
    by_numa = {}
    for gpu in gpus:
        by_numa.setdefault(node.numa_of_gpu.get(gpu, 0), []).append(gpu)
    best = None
    for group in by_numa.values():
        if len(group) >= count and (best is None or len(group) > len(best)):
            best = group
    return best


def _candidate_nodes(job, nodes):
    candidates = []
    for node in nodes:
        if node.free_gpus <= 0:
            continue
        if job.gpu_model and node.gpu_model != job.gpu_model:
            continue
        if node.gpu_memory_gb < job.min_gpu_memory_gb:
            continue
        candidates.append(node)
    return candidates


def _try_single_node(job, candidates, used, forbidden_domains):
    """Best-fit single-node placement; returns (node, gpus) or None."""
    fits = []
    for node in candidates:
        if node.fault_domain in forbidden_domains:
            continue
        free = set(range(node.gpu_count)) - used[node.name]
        gpus = _select_gpus(node, free, job.gpus_per_replica,
                            job.min_link, job.numa_local)
        if gpus is not None:
            fits.append((len(free) - job.gpus_per_replica, node, gpus))
    if not fits:
        return None
    fits.sort(key=lambda item: (item[0], item[1].name))
    _, node, gpus = fits[0]
    return node, gpus


def _try_multi_node(job, candidates, used, forbidden_domains,
                    single_fault_domain):
    """Spread one replica across nodes within the allowed switch tier.

    Returns (tier, [(node, gpus), ...]) or None. When
    ``single_fault_domain`` is set the whole replica must stay inside
    one fault domain.
    """
    for tier in ("leaf", "spine"):
        if not link_satisfies(tier, job.min_link):
            continue
        groups = {}
        for node in candidates:
            if node.fault_domain in forbidden_domains:
                continue
            key = getattr(node, tier)
            if single_fault_domain:
                key = (key, node.fault_domain)
            groups.setdefault(key, []).append(node)
        for key in sorted(groups):
            group = sorted(groups[key],
                           key=lambda n: (-(n.free_gpus - len(used[n.name])), n.name))
            picked = []
            total = 0
            for node in group:
                free = set(range(node.gpu_count)) - used[node.name]
                if not free:
                    continue
                take = min(len(free), job.gpus_per_replica - total)
                gpus = _select_gpus(node, free, take, "pcie", job.numa_local)
                if gpus is None:
                    continue
                picked.append((node, gpus))
                total += len(gpus)
                if total >= job.gpus_per_replica:
                    return tier, picked
    return None


def _replica_bottleneck(node_by_name, allocations, cross_tier):
    levels = []
    for node_name, gpus in allocations:
        levels.append(_intra_node_link(node_by_name[node_name], gpus))
    if len(allocations) > 1:
        levels.append(cross_tier)
    return weakest(levels)


def plan(job, nodes):
    """Compute a placement for ``job`` on ``nodes``.

    Returns a :class:`PlacementResult`; on failure ``reasons`` explains
    why no combination satisfies the link level and fault-domain rules.
    """
    result = PlacementResult(job_name=job.name, success=False)
    node_by_name = {node.name: node for node in nodes}
    candidates = _candidate_nodes(job, nodes)
    if not candidates:
        result.reasons.append(
            "no node matches gpu_model=%r and min_gpu_memory_gb=%s with free GPUs"
            % (job.gpu_model, job.min_gpu_memory_gb))
        return result

    total_free = sum(node.free_gpus for node in candidates)
    total_needed = job.replicas * job.gpus_per_replica
    if total_free < total_needed:
        result.reasons.append(
            "not enough free GPUs: need %d, only %d free on matching nodes"
            % (total_needed, total_free))
        return result

    if job.fault_domain_aware:
        domains = {node.fault_domain for node in candidates}
        if len(domains) < job.replicas:
            result.reasons.append(
                "fault-domain limit: %d replicas requested but only %d distinct "
                "fault domains available (%s)"
                % (job.replicas, len(domains), ", ".join(sorted(domains))))
            return result

    # Occupied GPUs (from other jobs) are modelled as already-used indices.
    used = {}  # type: Dict[str, Set[int]]
    for node in nodes:
        occupied = node.gpu_count - node.free_gpus
        used[node.name] = set(range(occupied))
    used_domains = set()  # type: Set[str]
    reasons = []

    for replica in range(job.replicas):
        forbidden = set(used_domains) if job.fault_domain_aware else set()
        placement = _try_single_node(job, candidates, used, forbidden)
        if placement is not None:
            node, gpus = placement
            allocations = [(node.name, gpus)]
            bottleneck = _replica_bottleneck(node_by_name, allocations, None)
            domain = node.fault_domain
        else:
            multi = None
            if link_rank(job.min_link) >= link_rank("leaf"):
                multi = _try_multi_node(job, candidates, used, forbidden,
                                        job.fault_domain_aware)
            if multi is None:
                reasons.append(
                    "replica %d: no node combination satisfies gpus=%d, "
                    "min_link=%s, numa_local=%s within one fault domain"
                    % (replica, job.gpus_per_replica, job.min_link,
                       job.numa_local))
                break
            tier, picked = multi
            allocations = [(node.name, gpus) for node, gpus in picked]
            bottleneck = _replica_bottleneck(node_by_name, allocations, tier)
            domain = picked[0][0].fault_domain

        for node_name, gpus in allocations:
            used[node_name].update(gpus)
        used_domains.add(domain)
        result.assignments.append(ReplicaPlacement(
            replica=replica,
            allocations=allocations,
            bottleneck=bottleneck,
            fault_domain=domain,
        ))

    if len(result.assignments) < job.replicas:
        result.reasons.extend(reasons)
        result.assignments = []
        return result

    result.success = True
    result.bottleneck = weakest(p.bottleneck for p in result.assignments)
    for node in candidates:
        allocated = len(used[node.name]) - (node.gpu_count - node.free_gpus)
        remaining = node.free_gpus - allocated
        if allocated > 0 and remaining > 0:
            result.fragments[node.name] = remaining
    return result
