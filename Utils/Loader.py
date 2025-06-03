import numpy as np
from SGRLvCLT import *


class Loader:
    def __init__(self):
        self.countID = 0
        self.G = {}
        self.co = {}
        self.revco = {}
        self.nodeFeatures = []
        self.digit = 4

    def nodeID(self, l, x1, f):
        key = f"{l}:{x1}"
        if key not in self.co:
            self.co[key] = self.countID
            self.revco[self.countID] = key
            self.nodeFeatures.append(f[x1])
            self.countID += 1
        return self.co[key]

    def read(self, file, nodeFeaturesMap, l=""):
        x = file.values
        for a in range(x.shape[0]):
            i = self.nodeID(str(l), str(x[a, 0]), nodeFeaturesMap)
            j = self.nodeID(str(l), str(x[a, 1]), nodeFeaturesMap)
            self.addEdge((i, j), x[a, 2:])

    def readUnion(self, file_list):
        for idx, (file, nodeFeaturesMap) in enumerate(file_list):
            self.read(file, nodeFeaturesMap, l=idx)

    def storeEmb(self, path, data):
        with open(path, 'w') as f:
            for a in range(data.shape[0]):
                row = f"{self.revco[a]} " + " ".join(map(str, data[a]))
                f.write(row + "\n")

    def addEdge(self, s, f):
        l1, l2 = s
        for node in [l1, l2]:
            if node not in self.G:
                self.G[node] = {'in': {}, 'out': {}}
        self.G[l1]['out'][l2] = f
        self.G[l2]['in'][l1] = f

    def explore(self, a, lev):
        visited = set()
        queue = [(a, 0)]
        visited.add(a)
        while queue:
            m, l = queue.pop(0)
            if l < lev:
                for neighbor in self.G[m]['out']:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, l + 1))
        return visited

    # --- Finalization Variants ---
    def _normalize(self, lis, node_out, transform_func):
        hh = transform_func(lis)
        G1 = {a: {'out': {}} for a in node_out}
        for a in node_out:
            count = 0
            for b in self.G[a]['out']:
                features = hh[count].reshape(-1)
                G1[a]['out'][b] = np.hstack([features, self.G[a]['out'][b]])
                count += 1
        self.G = G1
        self.nodeFeatures = np.vstack(self.nodeFeatures)
        self.nodeFeatures = self.nodeFeatures[:, 3:]

    def finalNormalized(self):
        node_out = {}
        for a in self.G:
            node_out[a] = [self.nodeFeatures[a][:3].reshape(3, 1)]
            for b in self.G[a]['out']:
                node_out[a].append(self.nodeFeatures[b][:3].reshape(3, 1))
        self._normalize(node_out, self.G, standardTotal)

    def finalNormalizedDistance(self):
        self._add_distance_features(standardTotal, keep_node_features=False)

    def finalNormalizedDistanceNF(self):
        self._add_distance_features(standardTotal, keep_node_features=False, drop_extra=True)

    def finalNormalizedDistancePosition(self):
        self._add_distance_features(standardTotal, keep_node_features=True)

    def finalNormalizedDistancePositionNF(self):
        self._add_distance_features(standardTotal, keep_node_features=True, drop_extra=True)

    def finalNormalizedPosition(self):
        self._add_features_only(standardTotal, keep_node_features=True)

    def finalNormalizedPositionNF(self):
        self._add_features_only(standardTotal, keep_node_features=True, drop_extra=True)

    def finalDistance(self):
        self._add_distance_only(drop_extra=True)

    def finalDistanceNF(self):
        self._add_distance_only(drop_extra=True, override_node_features=True)

    def finalPosition(self):
        self.nodeFeatures = np.vstack(self.nodeFeatures)

    def finalPositionNF(self):
        self.nodeFeatures = np.vstack(self.nodeFeatures)
        self.nodeFeatures = self.nodeFeatures[:, 0:3]
        for a in self.G:
            self.G[a] = {'out': {b: np.array([0]) for b in self.G[a]['out']}}

    def finalPositionDistance(self):
        self._add_distance_only(keep_position=True)

    def finalPositionDistanceNF(self):
        self._add_distance_only(keep_position=True, drop_extra=True)

    # --- Internal helpers ---
    def _add_distance_features(self, transform_func, keep_node_features=False, drop_extra=False):
        G1 = {}
        for a in self.G:
            G1[a] = {'out': {}}
            lis = [self.nodeFeatures[a][:3].reshape(3, 1)]
            for b in self.G[a]['out']:
                lis.append(self.nodeFeatures[b][:3].reshape(3, 1))
            hh = transform_func(lis)
            count = 0
            for b in self.G[a]['out']:
                dist = np.linalg.norm(self.nodeFeatures[a][:3] - self.nodeFeatures[b][:3])
                dist = round(dist, self.digit)
                G1[a]['out'][b] = np.hstack([np.array([dist]), hh[count].reshape(3,), self.G[a]['out'][b]])
                count += 1
        self.G = G1
        self.nodeFeatures = np.vstack(self.nodeFeatures)
        if drop_extra:
            self.nodeFeatures = np.zeros((self.nodeFeatures.shape[0], 1))
        elif not keep_node_features:
            self.nodeFeatures = self.nodeFeatures[:, 3:]

    def _add_features_only(self, transform_func, keep_node_features=False, drop_extra=False):
        G1 = {}
        for a in self.G:
            G1[a] = {'out': {}}
            lis = [self.nodeFeatures[a][:3].reshape(3, 1)]
            for b in self.G[a]['out']:
                lis.append(self.nodeFeatures[b][:3].reshape(3, 1))
            hh = transform_func(lis)
            count = 0
            for b in self.G[a]['out']:
                G1[a]['out'][b] = hh[count].reshape(3,)
                count += 1
        self.G = G1
        self.nodeFeatures = np.vstack(self.nodeFeatures)
        if drop_extra:
            self.nodeFeatures = np.zeros((self.nodeFeatures.shape[0], 1))
        elif keep_node_features:
            self.nodeFeatures = self.nodeFeatures[:, 0:3]

    def _add_distance_only(self, keep_position=False, drop_extra=False, override_node_features=False):
        G1 = {}
        for a in self.G:
            G1[a] = {'out': {}}
            for b in self.G[a]['out']:
                dist = np.linalg.norm(self.nodeFeatures[a][:3] - self.nodeFeatures[b][:3])
                dist = round(dist, self.digit)
                G1[a]['out'][b] = np.array([dist])
        self.G = G1
        self.nodeFeatures = np.vstack(self.nodeFeatures)
        if override_node_features:
            self.nodeFeatures = np.zeros((self.nodeFeatures.shape[0], 1))
        elif keep_position:
            self.nodeFeatures = self.nodeFeatures[:, 0:3]
        elif drop_extra:
            self.nodeFeatures = self.nodeFeatures[:, 3:]
