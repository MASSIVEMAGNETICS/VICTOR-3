import unittest

from victor_poc import DEFAULT_ACTIVATION_THRESHOLD, VictorSystem


class _Clock:
    def __init__(self, start: float = 0.0):
        self.current = start

    def tick(self) -> float:
        self.current += 1.0
        return self.current


class VictorPocTests(unittest.TestCase):
    def test_end_to_end_flow_builds_causal_links_and_world_state(self):
        clock = _Clock()
        system = VictorSystem(time_provider=clock.tick)

        first = system.process("engine warms because fuel ignites")
        second = system.process("engine is hot")

        # Memory timestamps follow the deterministic clock.
        self.assertEqual(first["memory"].timestamp, 1.0)
        self.assertEqual(second["memory"].timestamp, 2.0)

        # Causal chain connects sequential observations.
        self.assertEqual(second["reasoning"]["causal_links"], [("engine warms because fuel ignites", "engine is hot")])

        # Activation spreads to multiple nodes on the grid.
        active_nodes = [v for v in second["activation"].values() if v > DEFAULT_ACTIVATION_THRESHOLD]
        self.assertGreater(len(active_nodes), 1)

        # World model retains parsed facts.
        self.assertIn("engine=hot", second["world"])


if __name__ == "__main__":
    unittest.main()
