from scipy import linalg
from scipy.optimize import minimize
import numpy as np


def signal(X1norm, eigen):
    f = np.matmul(X1norm, eigen)
    f[np.abs(f) <= 1e-10] = 0
    if f[0, 0] < 0:
        return -f
    else:
        if f[0, 0] > 0:
            return f
        else:
            vp = np.max(f[1:, 0])
            vn = np.max(-f[1:, 0])
            if vp > vn:
                return f
            else:
                if vp < vn:
                    return -f
                elif vp == 0 or vn == 0:
                    return np.zeros(f.shape)
                else:
                    return np.zeros(f.shape)


def signal1(X1norm, eigens):
    f = np.matmul(X1norm, eigens)
    h = [[0]]
    lis = []
    best = -1e9

    for i in range(1, f.shape[0]):
        fun = lambda x: -np.sum(np.matmul(x.reshape(1, eigens.shape[1]), f[i, :].reshape(f.shape[1], 1)))
        cons = (
            {'type': 'ineq', 'fun': lambda x: np.sum(np.matmul(x.reshape(1, eigens.shape[1]), f[0, :].reshape(f.shape[1], 1)))},
            {'type': 'ineq', 'fun': lambda x: -np.sum(np.power(np.matmul(x.reshape(1, eigens.shape[1]), eigens.T), 2)) + 1}
        )
        bnds = [(None, None) for _ in range(eigens.shape[1])]
        res = minimize(fun, x0=np.sqrt(1 / eigens.shape[1]) * np.ones(eigens.shape[1]), method='SLSQP', bounds=bnds, constraints=cons)
        val = np.round(-res.fun, 10)
        if val >= best:
            if -res.fun == best:
                lis.append(i)
            else:
                lis = [i]
                best = val

    f1 = np.matmul(np.mean(X1norm[lis, :], axis=0).reshape(1, X1norm.shape[1]), eigens)

    for i in range(1, f.shape[0]):
        fun = lambda x: -np.sum(np.matmul(x.reshape(1, eigens.shape[1]), f[i, :].reshape(f.shape[1], 1)))
        cons = (
            {'type': 'ineq', 'fun': lambda x: np.sum(np.matmul(x.reshape(1, eigens.shape[1]), f[0, :].reshape(f.shape[1], 1)))},
            {'type': 'ineq', 'fun': lambda x: np.sum(np.matmul(x.reshape(1, eigens.shape[1]), f1.reshape(f.shape[1], 1))) - best},
            {'type': 'ineq', 'fun': lambda x: -np.sum(np.power(np.matmul(x.reshape(1, eigens.shape[1]), eigens.T), 2)) + 1}
        )
        bnds = [(None, None) for _ in range(eigens.shape[1])]
        res = minimize(fun, x0=np.sqrt(1 / eigens.shape[1]) * np.ones(eigens.shape[1]), method='SLSQP', bounds=bnds, constraints=cons)
        h.append([-res.fun])

    return np.array(h)


total_indic = []


def standardTotal(P, size=3, precision=10):
    X1 = np.vstack([x.transpose() for x in P])
    X1norm = X1 - np.mean(X1, axis=0)
    A = np.cov(X1norm.transpose())
    w, z = linalg.eig(A)
    w1 = np.round(w, precision)

    dic = {}
    opt_active = []

    for x1 in range(w1.shape[0]):
        x = -w1[x1]
        if x not in dic:
            dic[x] = [0, []]
        dic[x][0] += 1
        dic[x][1].append(x1)
        if np.round(x, precision) != 0:
            opt_active.append(x1)

    for s in dic:
        dic[s][1] = np.array(dic[s][1])

    ll = np.sort(np.array(list(dic.keys())))
    hh = []

    for s in ll:
        sa = dic[s]
        if sa[1].shape[0] == 1:
            hh.append(signal(X1norm, z[:, sa[1][0]].reshape((z.shape[0], 1))))
        else:
            if np.round(s, precision) != 0:
                hh.append(signal1(X1norm, z[:, sa[1]].reshape((z.shape[0], sa[1].shape[0]))))

    if len(hh) == 0:
        H = np.zeros((X1.shape[0], size))
    else:
        ff = np.hstack(hh)
        H = np.round(ff.real, precision)

    l = []
    bol = True

    for p in H:
        if bol:
            bol = False
        else:
            l.append(np.array([p[i] if i < H.shape[1] else 0 for i in range(size)]).reshape((size, 1)))

    total_indic.append(opt_active)
    return l
