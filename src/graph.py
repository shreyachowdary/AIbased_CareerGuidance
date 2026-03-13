"""
Graph data structure and algorithms - internal implementation.
Uses adjacency list representation. Implements BFS, DFS, and shortest path.
"""

from collections import deque
from typing import Dict, List, Optional, Set


class Graph:
    """
    Directed graph using adjacency list representation.
    Nodes are stored as keys; edges as lists of neighbors.
    """

    def __init__(self, directed: bool = True):
        self._adj: Dict[str, List[str]] = {}
        self._directed = directed

    def add_node(self, node: str) -> None:
        """Add a node if not present."""
        if node not in self._adj:
            self._adj[node] = []

    def add_edge(self, u: str, v: str) -> None:
        """Add edge from u to v."""
        self.add_node(u)
        self.add_node(v)
        if v not in self._adj[u]:
            self._adj[u].append(v)
        if not self._directed and u not in self._adj[v]:
            self._adj[v].append(u)

    def nodes(self) -> List[str]:
        """Return all nodes."""
        return list(self._adj.keys())

    def neighbors(self, node: str) -> List[str]:
        """Return neighbors of a node."""
        return self._adj.get(node, [])

    def num_nodes(self) -> int:
        return len(self._adj)

    def num_edges(self) -> int:
        return sum(len(neighbors) for neighbors in self._adj.values())

    def bfs(self, start: str) -> List[str]:
        """
        Breadth-First Search from start node.
        Returns nodes in order of visitation (level order).
        """
        if start not in self._adj:
            return []
        visited: Set[str] = set()
        result: List[str] = []
        queue: deque = deque([start])
        visited.add(start)

        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in self._adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return result

    def dfs(self, start: str) -> List[str]:
        """
        Depth-First Search from start node (iterative, using stack).
        Returns nodes in order of visitation.
        """
        if start not in self._adj:
            return []
        visited: Set[str] = set()
        result: List[str] = []
        stack: List[str] = [start]

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            result.append(node)
            for neighbor in reversed(self._adj.get(node, [])):
                if neighbor not in visited:
                    stack.append(neighbor)

        return result

    def shortest_path_bfs(self, start: str, end: str) -> Optional[List[str]]:
        """
        Shortest path from start to end using BFS (unweighted).
        Returns path as list of nodes, or None if no path exists.
        """
        if start not in self._adj or end not in self._adj:
            return None
        if start == end:
            return [start]

        parent: Dict[str, Optional[str]] = {start: None}
        queue: deque = deque([start])

        while queue:
            node = queue.popleft()
            for neighbor in self._adj.get(node, []):
                if neighbor not in parent:
                    parent[neighbor] = node
                    if neighbor == end:
                        path = []
                        curr = end
                        while curr is not None:
                            path.append(curr)
                            curr = parent[curr]
                        return path[::-1]
                    queue.append(neighbor)

        return None
