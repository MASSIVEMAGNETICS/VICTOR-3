from __future__ import annotations

import heapq
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple


Coordinate = Tuple[int, int]
DEFAULT_ACTIVATION_THRESHOLD = 0.5


def _now() -> float:
    return time.time()


class FlowerOfLifeGrid:
    """
    Lightweight axial-grid approximation of the Flower of Life geometry.

    The grid is intentionally tiny to stay well below the 1GB footprint while
    still providing stable addressing for sparse activations.
    """

    # Axial coordinates for a hex grid; used to approximate the Flower of Life layout.
    _DIRECTIONS: Tuple[Coordinate, ...] = (
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
    )

    def __init__(self, radius: int = 2):
        self.radius = radius
        self.nodes: Set[Coordinate] = self._build_nodes()

    def _build_nodes(self) -> Set[Coordinate]:
        nodes: Set[Coordinate] = set()
        for q in range(-self.radius, self.radius + 1):
            for r in range(-self.radius, self.radius + 1):
                if abs(q + r) <= self.radius:
                    nodes.add((q, r))
        return nodes

    def neighbors(self, coord: Coordinate) -> List[Coordinate]:
        return [
            (coord[0] + dq, coord[1] + dr)
            for dq, dr in self._DIRECTIONS
            if (coord[0] + dq, coord[1] + dr) in self.nodes
        ]


class LightPropagator:
    """
    Spreads activation along the Flower of Life grid to simulate recall.
    """

    def __init__(self, grid: FlowerOfLifeGrid, decay: float = 0.6):
        self.grid = grid
        self.decay = decay

    def propagate(self, seeds: Iterable[Coordinate], steps: int = 2) -> Dict[Coordinate, float]:
        activation: Dict[Coordinate, float] = {}
        frontier: Set[Coordinate] = {coord for coord in seeds if coord in self.grid.nodes}
        for coord in frontier:
            activation[coord] = 1.0

        for _ in range(steps):
            next_frontier: Set[Coordinate] = set()
            for coord in frontier:
                base = activation.get(coord, 1.0) * self.decay
                for neighbor in self.grid.neighbors(coord):
                    new_val = max(base, activation.get(neighbor, 0.0))
                    if new_val > activation.get(neighbor, 0.0):
                        activation[neighbor] = new_val
                        next_frontier.add(neighbor)
            frontier = next_frontier
        return activation


@dataclass
class SDRSnapshot:
    text: str
    sdr: Set[int]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SDRMemory:
    """
    Deterministic SDR encoder so tests can assert on stored snapshots.
    """

    def __init__(self, size: int = 256, active_bits: int = 12, time_provider: Optional[Callable[[], float]] = None):
        self.size = size
        self.active_bits = active_bits
        self.snapshots: List[SDRSnapshot] = []
        self.time_provider = time_provider or _now
        if self.active_bits > self.size:
            raise ValueError("active_bits must be less than or equal to size")

    def _encode(self, text: str) -> Set[int]:
        rng = random.Random(hash(text))
        return set(rng.sample(range(self.size), self.active_bits))

    def remember(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> SDRSnapshot:
        snapshot = SDRSnapshot(
            text=text,
            sdr=self._encode(text),
            timestamp=self.time_provider(),
            metadata=metadata or {},
        )
        self.snapshots.append(snapshot)
        return snapshot

    def to_coordinates(self, grid: FlowerOfLifeGrid, snapshot: SDRSnapshot) -> List[Coordinate]:
        nodes = sorted(grid.nodes)
        if not nodes:
            return []
        coords: List[Coordinate] = []
        for bit in sorted(snapshot.sdr):
            coords.append(nodes[bit % len(nodes)])
        return coords


class CausalReasoner:
    """
    Builds a lightweight causal chain from sequential observations.
    """

    def __init__(self):
        self.links: List[Tuple[str, str]] = []
        self.last_observation: Optional[str] = None

    def observe(self, text: str) -> List[Tuple[str, str]]:
        if self.last_observation:
            self.links.append((self.last_observation, text))
        self.last_observation = text
        return list(self.links)

    def predict_next(self, text: str) -> str:
        # Naive heuristic: return the last observed effect linked to the same cause,
        # ignoring frequencies and probability distribution.
        matching = [effect for cause, effect in self.links if cause == text]
        return matching[-1] if matching else ""


class WorldModel:
    """
    Maintains a compact, time-stamped belief store that can be pruned.
    """

    def __init__(self, max_entries: int = 128, time_provider: Optional[Callable[[], float]] = None):
        self.max_entries = max_entries
        self.time_provider = time_provider or _now
        self.state: Dict[str, float] = {}

    def update(self, facts: Iterable[str], timestamp: Optional[float] = None) -> Dict[str, float]:
        current_time = timestamp if timestamp is not None else self.time_provider()
        for fact in facts:
            self.state[fact] = current_time
        # Prune oldest entries if storage exceeds maximum capacity.
        overflow = len(self.state) - self.max_entries
        if overflow > 0:
            oldest = heapq.nsmallest(overflow, self.state.items(), key=lambda item: item[1])
            for fact, _ in oldest:
                self.state.pop(fact, None)
        return dict(self.state)

    def snapshot(self) -> Dict[str, float]:
        return dict(self.state)


class VictorSystem:
    """
    End-to-end proof-of-concept pipeline wiring the roadmap components.
    """

    def __init__(
        self,
        *,
        grid_radius: int = 2,
        decay: float = 0.6,
        time_provider: Optional[Callable[[], float]] = None,
    ):
        self.grid = FlowerOfLifeGrid(radius=grid_radius)
        provider = time_provider or _now
        self.memory = SDRMemory(time_provider=provider)
        self.light = LightPropagator(self.grid, decay=decay)
        self.reasoner = CausalReasoner()
        self.world = WorldModel(time_provider=provider)

    def process(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        snapshot = self.memory.remember(text, metadata)
        coords = self.memory.to_coordinates(self.grid, snapshot)
        activation = self.light.propagate(coords)

        causal_links = self.reasoner.observe(text)
        prediction = self.reasoner.predict_next(text)

        facts = self._extract_facts(text)
        world_view = self.world.update(facts, timestamp=snapshot.timestamp)

        return {
            "memory": snapshot,
            "activation": activation,
            "reasoning": {"causal_links": causal_links, "prediction": prediction},
            "world": world_view,
        }

    @staticmethod
    def _extract_facts(text: str) -> List[str]:
        """
        Minimal fact extractor: uses the first matching delimiter (one of
        ' is ', ' are ', ' =', ':') to build a single key=value style fact;
        otherwise returns the lower-cased text.
        """
        lowered = text.lower()
        for delimiter in (" is ", " are ", " =", ":"):
            if delimiter in lowered:
                parts = [p.strip() for p in lowered.split(delimiter) if p.strip()]
                if len(parts) >= 2:
                    return [f"{a}={b}" for a, b in zip(parts[:-1], parts[1:])]
        return [lowered]


def demo() -> None:
    victor = VictorSystem()
    interactions = [
        "Engine warms because fuel ignites",
        "Engine is hot",
        "Cooling is required",
    ]
    activation_threshold = DEFAULT_ACTIVATION_THRESHOLD
    for message in interactions:
        result = victor.process(message)
        print(f"\nInput: {message}")
        print(f"Memory timestamp: {result['memory'].timestamp}")
        print(f"Activation hotspots: {len([v for v in result['activation'].values() if v > activation_threshold])}")
        print(f"Causal links: {result['reasoning']['causal_links']}")
        print(f"Prediction hint: {result['reasoning']['prediction']!r}")
        print(f"World facts: {list(result['world'].keys())}")


if __name__ == "__main__":
    demo()
