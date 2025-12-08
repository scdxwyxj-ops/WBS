"""Minimal undirected graph fallback when NetworkX is unavailable."""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Set


class SimpleGraph:
    def __init__(self) -> None:
        self._adjacency: Dict[int, Set[int]] = {}

    # NetworkX-compatible API subset -------------------------------------
    def add_node(self, node: int) -> None:
        self._adjacency.setdefault(int(node), set())

    def add_edge(self, u: int, v: int) -> None:
        u = int(u)
        v = int(v)
        self.add_node(u)
        self.add_node(v)
        self._adjacency[u].add(v)
        self._adjacency[v].add(u)

    def has_node(self, node: int) -> bool:
        return int(node) in self._adjacency

    def is_directed(self) -> bool:
        return False

    def neighbors(self, node: int) -> Iterable[int]:
        return self._adjacency.get(int(node), ())

    def successors(self, node: int) -> Iterable[int]:
        return self.neighbors(node)

    def predecessors(self, node: int) -> Iterable[int]:
        return self.neighbors(node)

    def nodes(self) -> Iterator[int]:
        return iter(self._adjacency.keys())
