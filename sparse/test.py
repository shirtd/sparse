from util import *

def test_perm(G):
    for i, x in enumerate(G.P):
        res = True
        if i > 0:
            mn = min(la.norm(x - y) for y in G.P[:i+1])
            mx = max(min(la.norm(p - q) for q in G.P[:i]) for p in G.P)
            if round(1e5 * mn) != round(1e5 * mx):
                print('%0.4f\t%0.4f' % (mn, mx))
                res = False
    return res

def test_pred(G):
    res = True
    for i, p in enumerate(G.P):
        if i > 0:
            L, d = G.L[i], la.norm(p - G.P[G.pred(i)])
            if round(1e4 * L) != round(1e4 * d):
                print('%0.4f\t%0.4f' % (L, d))
                res = False
    return res

def test_parent(G, i):
    res = True
    for j, p in enumerate(G.P[:i+1]):
        if G.l[i] < G.l[j] and G.par[j] != j:
            print('l(%d) < l(%d) and par(%d) != %d' % (i, j, j, j))
            res = False
        elif not (G.l[G.par[j]] > G.l[i] and G.d[j, G.par[j]] <= 2 ** G.l[i]):
            print('l(par(%d)) <= l(%d) or d(p%d, p%d) > 2 ** l(%d)' % (G.par[j], i, j, G.par[j], i))
            res = False
    return res

def test_child(G, i):
    res = True
    for j, p in enumerate(G.P[:i+1]):
        K = [k for k in range(i+1) if G.par[k] == j and G.l[k] == G.l[i]]
        if not (j in G.chl[j] and all(k in G.chl[j] for k in K)):
            print("ch(%d) does not contain {p%d} U {pk in P%d | par(k) "
                    "= %d and l(k) = l(%d)}" % (j, j, i, j, i))
            res = False
    return res

def test_nbr(G, i):
    res = True
    for j, p in enumerate(G.P[:i+1]):
        fmin = lambda k: min((G.l[j], G.l[k], G.l[i] + 1))
        K = [k for k in range(i+1) if G.d[j, k] <= G.kap * 2 ** fmin(k)]
        if not all(k in G.nbr[j] for k in K):
            print("nbr(%d) does not contain {pk in P%d | d(p%d, pk) <= kap "
                    "* 2 ** min(l(%d), l(k), l(%d) + 1) " % (j, i, j, j, i))
            res = False
    return res

def test_orient(R, dim=2):
    for s, w in R.S[dim]:
        for i in range(1, len(s)):
            e = (s[i-1], s[i])
            if not e in R.E:
                print(e, s)
