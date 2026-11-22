"""Rooted-tree and path-system primitives underlying neighborly path system enumeration.

Lifted near-verbatim from metrizable_wheels/utils/PathAndTreeSystems.py, which was
already the cleanest, most general implementation of this data model in the repo.
"""
from collections.abc import MutableMapping
from itertools import combinations
from copy import deepcopy


class RootedTree:
    """A rooted tree built incrementally by attaching leaves to existing nodes."""

    def __init__(self, root):
        self.root = root
        self.nodes = set()
        self.nodes.add(root)
        self.successors_dict = {root: set()}
        self.predecessor_dict = {root: None}

    def addNode(self, u, v):
        """Attach whichever of u, v is not yet in the tree to the one that is."""
        if u in self.nodes and v not in self.nodes:
            self.nodes.add(v)
            self.successors_dict[v] = set()
            self.predecessor_dict[v] = u
            self.successors_dict[u].add(v)
        elif v in self.nodes and u not in self.nodes:
            self.nodes.add(u)
            self.successors_dict[u] = set()
            self.predecessor_dict[u] = v
            self.successors_dict[v].add(u)
        else:
            raise ValueError(f"Tree cannot be extended with edge ({u}, {v})")

    def predecessor(self, u):
        return self.predecessor_dict[u]

    def successors(self, u):
        return self.successors_dict[u]


def orderPath(p):
    """Orient a path/pair so the smaller endpoint comes first, e.g. (3,2,1) -> (1,2,3)."""
    if p[0] < p[-1]:
        return p
    return p[::-1]


def getSubpaths(path):
    """Return {(u,v): subpath} for every pair of vertices along `path`."""
    path = orderPath(path)
    subPathDict = {}
    n = len(path)
    for i, j in combinations(range(n), 2):
        pair = orderPath((path[i], path[j]))
        sub = tuple(orderPath(path[i:j + 1]))
        subPathDict[pair] = sub
    return subPathDict


class InconsistentPathError(Exception):
    """Raised when adding a path to a consistent PathSystem would break consistency."""


class PathSystem(MutableMapping):
    """A dict from vertex pairs to the path connecting them, keyed order-independently.

    When constructed with consistent=True, every assignment is checked against all
    subpaths already present and raises InconsistentPathError on conflict.
    """

    def __init__(self, consistent=False, data=()):
        self.mapping = {}
        self.consistent = consistent
        self.update(data)

    def __getitem__(self, key):
        u, v = key
        if u < v:
            return self.mapping[(u, v)]
        return self.mapping[(v, u)][::-1]

    def __delitem__(self, key):
        del self.mapping[key]

    def __setitem__(self, pair, path):
        pair = tuple(orderPath(pair))
        path = tuple(orderPath(path))

        if self.consistent:
            subPaths = getSubpaths(path)
            for p in set(subPaths.keys()) & set(self.mapping.keys()):
                if subPaths[p] != self.mapping[p]:
                    raise InconsistentPathError(f"{path} conflicts with existing {self.mapping[p]} on pair {p}")
            for p in set(subPaths.keys()) - set(self.mapping.keys()):
                self.mapping[p] = subPaths[p]
        else:
            self.mapping[pair] = path

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)

    def __repr__(self):
        return f"{type(self).__name__}({self.mapping})"

    def consistentAddition(self, path):
        """Check (without mutating) whether `path` could be added without conflict."""
        subPaths = getSubpaths(path)
        for pair in set(subPaths.keys()) & set(self.mapping.keys()):
            if subPaths[pair] != self.mapping[pair]:
                return False
        return True

    def add_path(self, path):
        pair = tuple(orderPath((path[0], path[-1])))
        self.__setitem__(pair, tuple(orderPath(path)))


def getTreeSystemFromPathSystem(pSystem):
    vtxs = set()
    for pair in pSystem:
        vtxs |= set(pSystem[pair])
    treeSystem = {}
    for v in vtxs:
        treeSystem[v] = RootedTree(v)
        for u in vtxs - {v}:
            path = pSystem[(v, u)]
            for i in range(len(path) - 1):
                if path[i + 1] not in treeSystem[v].nodes:
                    treeSystem[v].addNode(path[i], path[i + 1])
    return treeSystem


def isTreeSystemConsistent(treeSystem):
    tree = next(iter(treeSystem.values()))
    vtxs = list(tree.nodes)
    for u, v in combinations(vtxs, 2):
        uTree, vTree = treeSystem[u], treeSystem[v]
        path1 = [v]
        while path1[-1] != u:
            path1.append(uTree.predecessor(path1[-1]))
        path2 = [u]
        while path2[-1] != v:
            path2.append(vTree.predecessor(path2[-1]))
        if path1 != path2[::-1]:
            return False
    return True


def isPathSystemConsistent(pSystem):
    return isTreeSystemConsistent(getTreeSystemFromPathSystem(pSystem))
