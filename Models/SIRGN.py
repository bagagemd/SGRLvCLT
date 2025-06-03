from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np


class SirGN:
    # n: embedding size
    # n1: edge features
    # n2: graph features

    def __init__(self, n, n1, n2, levels=100):
        self.n = n
        self.n1 = n1
        self.n2 = n2
        self.leveles = levels
        self.levelProcess = []
        self.edgeFeatures = (MinMaxScaler(), KMeans(n_clusters=n1, random_state=1))
        self.graphRepresentation = (MinMaxScaler(), KMeans(n_clusters=n2, random_state=1))

    def getnumber(self, emb):
        ss = set()
        for x in range(emb.shape[0]):
            sd = ''
            for y in range(emb.shape[1]):
                sd += ',' + str(emb[x, y])
            ss.add(sd)
        return len(ss)

    def fit(self, G, fe):
        n = self.n
        n1 = self.n1
        nv = len(G)

        degree = np.array([[len(G[x]['out'])] for x in range(nv)])
        emb2 = np.hstack([degree, np.zeros((nv, n - 1))])
        emb = np.hstack([emb2, fe])

        count = self.getnumber(emb)
        b = []
        index = 0
        dicti = {}

        for o in range(nv):
            for j in G[o]['out']:
                f = G[o]['out'][j]
                b.append(f.reshape((1, f.shape[0])))
                dicti[str(o), "-", str(j)] = index
                index += 1

        b = np.vstack(b)
        scaler1 = self.edgeFeatures[0]
        embb = scaler1.fit_transform(b)
        kmeans1 = self.edgeFeatures[1].fit(embb)
        val1 = kmeans1.transform(embb)

        M1 = val1.max(axis=1)
        m1 = val1.min(axis=1)
        nv1 = b.shape[0]
        subx1 = (M1.reshape(nv1, 1) - val1) / (M1 - m1).reshape(nv1, 1)
        su1 = subx1.sum(axis=1)
        subx1 = subx1 / su1.reshape(nv1, 1)

        for i in range(self.leveles):
            proces = (MinMaxScaler(), KMeans(n_clusters=n, random_state=1))
            scaler = proces[0]
            emb3 = scaler.fit_transform(emb)
            kmeans = proces[1].fit(emb3)
            val = kmeans.transform(emb3)

            M = val.max(axis=1)
            m = val.min(axis=1)
            subx = (M.reshape(nv, 1) - val) / (M - m).reshape(nv, 1)
            su = subx.sum(axis=1)
            subx = subx / su.reshape(nv, 1)

            hh = np.zeros((nv, n * n1))
            for o in range(nv):
                for j in G[o]['out']:
                    index = dicti[str(j), "-", str(o)]
                    f = G[o]['out'][j]
                    hh[o, :] += (subx[j, :].reshape((n, 1)) * subx1[index, :]).flatten()

            emb5 = np.vstack(hh)
            emba = np.hstack([emb5, fe])
            d = self.getnumber(emba)

            if count >= d:
                break
            else:
                count = d
                emb = emba
                self.levelProcess.append(proces)

        graphemb = self.graphRepresentation[0].fit_transform(emb)
        self.graphRepresentation[1].fit(graphemb)
        return emb

    def transform(self, G, fe):
        n = self.n
        n1 = self.n1
        nv = len(G)

        degree = np.array([[len(G[x]['out'])] for x in range(nv)])
        emb2 = np.hstack([degree, np.zeros((nv, n - 1))])
        emb = np.hstack([emb2, fe])

        b = []
        index = 0
        dicti = {}

        for o in range(nv):
            for j in G[o]['out']:
                f = G[o]['out'][j]
                b.append(f.reshape((1, f.shape[0])))
                dicti[str(o), "-", str(j)] = index
                index += 1

        b = np.vstack(b)
        scaler1 = self.edgeFeatures[0]
        embb = scaler1.transform(b)
        kmeans1 = self.edgeFeatures[1]
        val1 = kmeans1.transform(embb)

        M1 = val1.max(axis=1)
        m1 = val1.min(axis=1)
        nv1 = b.shape[0]
        subx1 = (M1.reshape(nv1, 1) - val1) / (M1 - m1).reshape(nv1, 1)
        su1 = subx1.sum(axis=1)
        subx1 = subx1 / su1.reshape(nv1, 1)

        for i in range(len(self.levelProcess)):
            scaler = self.levelProcess[i][0]
            emb3 = scaler.transform(emb)
            kmeans = self.levelProcess[i][1]
            val = kmeans.transform(emb3)

            M = val.max(axis=1)
            m = val.min(axis=1)
            subx = (M.reshape(nv, 1) - val) / (M - m).reshape(nv, 1)
            su = subx.sum(axis=1)
            subx = subx / su.reshape(nv, 1)

            hh = np.zeros((nv, n * n1))
            for o in range(nv):
                for j in G[o]['out']:
                    index = dicti[str(j), "-", str(o)]
                    f = G[o]['out'][j]
                    hh[o, :] += (subx[j, :].reshape((n, 1)) * subx1[index, :]).flatten()

            emb5 = np.vstack(hh)
            emb = np.hstack([emb5, fe])

        return emb, subx1, dicti

    def transformGraph(self, G, fe):
        nv = len(G)
        emb, sss, dicti = self.transform(G, fe)
        graphemb = self.graphRepresentation[0].transform(emb)
        val1 = self.graphRepresentation[1].transform(graphemb)

        M1 = val1.max(axis=1)
        m1 = val1.min(axis=1)
        subx1 = (M1.reshape(nv, 1) - val1) / (M1 - m1).reshape(nv, 1)
        su1 = subx1.sum(axis=1)
        subx1 = subx1 / su1.reshape(nv, 1)

        n2 = self.n2
        return subx1.sum(axis=0).reshape((1, n2))

    def transformGraph0(self, G, fe):
        emb, sss, dicti = self.transform(G, fe)
        return emb.sum(axis=0).reshape((1, emb.shape[1]))

    def transformGraph1(self, G, fe):
        nv = len(G)
        n = self.n
        n1 = self.n1
        n2 = self.n2

        emb, sss, dicti = self.transform(G, fe)
        graphemb = self.graphRepresentation[0].transform(emb)
        val1 = self.graphRepresentation[1].transform(graphemb)

        M1 = val1.max(axis=1)
        m1 = val1.min(axis=1)
        subx1 = (M1.reshape(nv, 1) - val1) / (M1 - m1).reshape(nv, 1)
        su1 = subx1.sum(axis=1)
        subx1 = subx1 / su1.reshape(nv, 1)

        vec = np.zeros(n2 * n2 * n1)
        for o in range(nv):
            for j in G[o]['in']:
                index = dicti[str(o), "-", str(j)]
                vec += ((subx1[o].reshape((n2, 1)) * sss[index, :].reshape((1, n1))).flatten().reshape((n2 * n1, 1))
                        * subx1[j].reshape((1, n2))).flatten()

        return vec.reshape((1, vec.shape[0]))
