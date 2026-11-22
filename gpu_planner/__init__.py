"""Topology-aware GPU placement planner for distributed training jobs.

The planner only computes placement candidates. It never launches jobs.
"""

from .model import JobRequest, Node, LINK_LEVELS
from .planner import PlacementResult, ReplicaPlacement, plan

__all__ = [
    "JobRequest",
    "Node",
    "LINK_LEVELS",
    "PlacementResult",
    "ReplicaPlacement",
    "plan",
]
