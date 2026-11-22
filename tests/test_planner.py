import unittest

from gpu_planner import JobRequest, Node, plan


def nvlink_node(name, cliques, **kwargs):
    gpu_count = kwargs.pop("gpu_count", 8)
    return Node(
        name=name,
        gpu_model=kwargs.pop("gpu_model", "A100"),
        gpu_count=gpu_count,
        gpu_memory_gb=kwargs.pop("gpu_memory_gb", 80),
        nvlink_cliques=[set(c) for c in cliques],
        **kwargs
    )


class SingleNodeNvlinkTest(unittest.TestCase):
    def test_single_node_nvlink(self):
        node = nvlink_node("n1", [range(0, 4), range(4, 8)])
        job = JobRequest(name="job", replicas=1, gpus_per_replica=4,
                         min_link="nvlink")
        result = plan(job, [node])
        self.assertTrue(result.success, result.reasons)
        self.assertEqual(result.bottleneck, "nvlink")
        alloc = result.assignments[0].allocations
        self.assertEqual(len(alloc), 1)
        self.assertEqual(alloc[0][0], "n1")
        self.assertTrue(set(alloc[0][1]) <= set(range(0, 4)))

    def test_nvlink_required_but_clique_too_small(self):
        node = nvlink_node("n1", [range(0, 2), range(2, 4)])
        job = JobRequest(name="job", replicas=1, gpus_per_replica=4,
                         min_link="nvlink")
        result = plan(job, [node])
        self.assertFalse(result.success)
        self.assertTrue(any("min_link" in r for r in result.reasons))

    def test_pcie_fallback_when_clique_too_small(self):
        node = nvlink_node("n1", [range(0, 4), range(4, 8)])
        job = JobRequest(name="job", replicas=1, gpus_per_replica=8,
                         min_link="pcie")
        result = plan(job, [node])
        self.assertTrue(result.success, result.reasons)
        self.assertEqual(result.bottleneck, "pcie")


class CrossNodeTest(unittest.TestCase):
    def test_cross_node_pcie_fabric_same_leaf(self):
        nodes = [
            nvlink_node("n1", [range(0, 4)], gpu_count=4, leaf="leaf-1"),
            nvlink_node("n2", [range(0, 4)], gpu_count=4, leaf="leaf-1"),
        ]
        job = JobRequest(name="job", replicas=1, gpus_per_replica=8,
                         min_link="leaf")
        result = plan(job, nodes)
        self.assertTrue(result.success, result.reasons)
        self.assertEqual(result.bottleneck, "leaf")
        alloc = result.assignments[0].allocations
        self.assertEqual({name for name, _ in alloc}, {"n1", "n2"})

    def test_cross_node_blocked_when_pcie_required(self):
        nodes = [
            nvlink_node("n1", [range(0, 4)], gpu_count=4, leaf="leaf-1"),
            nvlink_node("n2", [range(0, 4)], gpu_count=4, leaf="leaf-1"),
        ]
        job = JobRequest(name="job", replicas=1, gpus_per_replica=8,
                         min_link="pcie")
        result = plan(job, nodes)
        self.assertFalse(result.success)
        self.assertTrue(any("min_link" in r for r in result.reasons))

    def test_cross_leaf_uses_spine(self):
        nodes = [
            nvlink_node("n1", [range(0, 4)], gpu_count=4,
                        leaf="leaf-1", spine="spine-1"),
            nvlink_node("n2", [range(0, 4)], gpu_count=4,
                        leaf="leaf-2", spine="spine-1"),
        ]
        job = JobRequest(name="job", replicas=1, gpus_per_replica=8,
                         min_link="spine")
        result = plan(job, nodes)
        self.assertTrue(result.success, result.reasons)
        self.assertEqual(result.bottleneck, "spine")


class FaultDomainTest(unittest.TestCase):
    def _nodes(self, count):
        return [
            nvlink_node("n%d" % idx, [range(0, 4)], gpu_count=4,
                        fault_domain="fd-%d" % idx)
            for idx in range(count)
        ]

    def test_replicas_spread_across_fault_domains(self):
        job = JobRequest(name="job", replicas=3, gpus_per_replica=2,
                         min_link="nvlink", fault_domain_aware=True)
        result = plan(job, self._nodes(3))
        self.assertTrue(result.success, result.reasons)
        domains = {p.fault_domain for p in result.assignments}
        self.assertEqual(len(domains), 3)

    def test_fault_domain_limit_rejects_job(self):
        job = JobRequest(name="job", replicas=4, gpus_per_replica=2,
                         min_link="nvlink", fault_domain_aware=True)
        result = plan(job, self._nodes(3))
        self.assertFalse(result.success)
        self.assertTrue(any("fault-domain" in r for r in result.reasons))


class FragmentTest(unittest.TestCase):
    def test_partially_free_node_used_and_fragment_recorded(self):
        node = nvlink_node("n1", [range(0, 4), range(4, 8)], free_gpus=6)
        job = JobRequest(name="job", replicas=1, gpus_per_replica=4,
                         min_link="pcie")
        result = plan(job, [node])
        self.assertTrue(result.success, result.reasons)
        self.assertEqual(result.fragments, {"n1": 2})

    def test_fragment_leftover_too_small_for_another_replica(self):
        nodes = [
            nvlink_node("n1", [range(0, 4), range(4, 8)], free_gpus=6),
            nvlink_node("n2", [range(0, 4), range(4, 8)], free_gpus=6),
        ]
        job = JobRequest(name="job", replicas=2, gpus_per_replica=4,
                         min_link="pcie")
        result = plan(job, nodes)
        self.assertTrue(result.success, result.reasons)
        # Each node donates 4 GPUs and keeps 2, too few for another
        # 4-GPU replica: both leftovers are recorded as fragments.
        self.assertEqual(result.fragments, {"n1": 2, "n2": 2})

    def test_not_enough_total_gpus(self):
        node = nvlink_node("n1", [range(0, 4)], gpu_count=4, free_gpus=2)
        job = JobRequest(name="job", replicas=1, gpus_per_replica=4,
                         min_link="pcie")
        result = plan(job, [node])
        self.assertFalse(result.success)
        self.assertTrue(any("not enough free GPUs" in r
                            for r in result.reasons))


if __name__ == "__main__":
    unittest.main()
