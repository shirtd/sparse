from sklearn.metrics.pairwise import pairwise_distances
from itertools import combinations
from functools import partial
import numpy.linalg as la
import dionysus as dio
from util import *

''''''''''''
''' UTIL '''
''''''''''''

def greedy_perm(data, verbose=True):
    d = pairwise_distances(data, metric='euclidean')
    perm = np.zeros(len(d), dtype=np.int64)
    lambdas = np.zeros(len(d), dtype=float)
    lambdas[0], ds = np.inf, d[0]
    frange = vrange(verbose, ' | order vertices ')
    for i in frange(1, len(d)):
        idx = np.argmax(ds)
        perm[i], lambdas[i] = idx, ds[idx]
        ds = np.minimum(ds, d[idx])
    return lambdas, perm

def _add_cofaces(E, L, e, adj, dim, N, t):
    w = max(E[p, q] for p in t for q in t if p > q) if len(t) > 1 else 0.
    l = min(L[i] * (1. + e) ** 2 / e for i in t)
    S = [(t, 2 * w)] if w < l else []
    if len(t) - 1 < dim:
        for v in N:
            M = filter(lambda w: w in N, adj[v])
            S = S + _add_cofaces(E, L, e, adj, dim, M, t + [v])
    return S

def add_cofaces(E, L, e, adj, dim, s):
    return _add_cofaces(E, L, e, adj, dim, adj[s], [s])

def fill_srips(data, dim=2, t=np.inf, eps=0.1):
    R = SparseRips(data, eps, t, dim)
    return R.incremental_vr(), R.data


''''''''''''''''''
''' CLASS DEFS '''
''''''''''''''''''

''''''''''''''''''
''' GRAPH BASE '''
class SparseStruct:
    def __init__(self, data, eps=0.1, verbose=True):
        self.n, data = len(data), data
        self.L, self.perm = greedy_perm(data, verbose)
        self.pmap = np.array(sorted(enumerate(self.perm), key=lambda (i, j): j))
        self.l, self.data = map(lambda x: np.ceil(np.log2(x)), self.L), data[self.perm]
        self.chl, self.par = {i : [] if i > 0 else [0] for i in range(self.n)}, {0 : 0}
        self.nbr = {i : set() if i > 0 else set([0]) for i in range(self.n)}
        self.kap, self.e = (eps ** 2 + 3. * eps + 2.) / eps, eps
        self.d = pairwise_distances(self.data, metric='euclidean')
    def dpar(self, i): return self.d[i, self.par[i]]
    def lkap(self, i): return self.kap * 2 ** self.l[i]
    def pred(self, i): return min(range(i), key=lambda j: self.d[i, j])
    def level(self, i): return filter(lambda j: self.l[j] == self.l[i], range(self.n))
    def init_level(self, i): map(lambda k: self.par.update({k : k}), self.level(i))
    def do_par(self, i, k): return self.d[i, k] <= self.dpar(i) and self.l[k] > self.l[i]
    def pred_par_nbr(self, i): return filter(lambda k: self.do_par(i, k), self.nbr[self.par[i]])
    def init_par(self, i): map(lambda k: self.par.update({i : k}), self.pred_par_nbr(i))
    def par_nbr_chl(self, i): return set(k for l in self.nbr[self.par[i]] for k in self.chl[l])
    def get_nbrs(self, i): return filter(lambda l: self.d[i, l] <= self.lkap(i), self.par_nbr_chl(i))
    def add_nbr(self, i, k): map(lambda (a, b): self.nbr[a].add(b), ((i,k), (k,i)))
    def insert(self, i):
        if i == 0: return
        if self.l[i] < self.l[i-1]:
            self.init_level(i-1)
        self.par[i] = self.par[self.pred(i)]
        self.init_par(i)
        self.nbr[i].add(i)
        self.chl[i].append(i)
        self.chl[self.par[i]].append(i)
        map(lambda k: self.add_nbr(i, k), self.get_nbrs(i))

''''''''''''''''''
''' GRAPH DEF  '''
class SparseGraph(SparseStruct):
    def __init__(self, data, eps=0.1, t=np.inf, verbose=True):
        SparseStruct.__init__(self, data, eps, verbose)
        self.E, self.t = {}, t
        self.adj = {i : [] for i in range(self.n)}
        self.construct_edges(verbose)
    def edge_birth(self, i, j):
        if i == j or self.t < self.d[i, j]: return np.inf
        i, j = (j, i) if self.L[i] > self.L[j] else (i, j)
        _e = (1. + self.e) / self.e
        if self.d[i, j] <= 2. * self.L[i] * _e:
            return self.d[i, j] / 2.
        if self.d[i, j] <= (self.L[i] + self.L[j]) * _e:
            return self.d[i, j] - self.L[i] * _e
        return np.inf
    def add_edge(self, i, j):
        a = self.edge_birth(i, j)
        if a < np.Inf:
            self.E[(i, j)] = a
            self.adj[i].append(j)
    def add_edges(self, i):
        map(lambda j: self.add_edge(i, j), self.nbr[i])
    def construct_edges(self, verbose):
        frange = vrange(verbose, ' | build graph ')
        for i in frange(self.n):
            self.insert(i)
            self.add_edges(i)

''''''''''''
''' RIPS '''
class SparseRips(SparseGraph):
    def __init__(self, data, eps=0.1, t=np.inf, dim=2, verbose=True):
        SparseGraph.__init__(self, data, eps, t, verbose)
        self.verbose = verbose
        self.dim = dim
    def partial(self, f):
        return partial(f, self.E, self.L, self.e, self.adj, self.dim)
    def incremental_vr(self, *args, **kw):
        fun = self.partial(add_cofaces)
        frange = vrange(self.verbose, ' | find simplices ')
        S = np.concatenate(tmap(fun, self.n, frange))
        fiter = viter(self.verbose, ' | make filtration ')
        F = dio.Filtration(map(lambda s: dio.Simplex(*s), fiter(S)))
        ftime = vtime(self.verbose, ' | sort filtration')
        ftime(F.sort)
        return F


''''''''''''''''''
''' PERSISTENCE'''
''''''''''''''''''
class Persistence:
    def __init__(self, F, verbose=True):
        self.F, self.H, self.D = F, {}, {}
        self.verbose = verbose
    def homology(self, prime=2):
        if self.verbose: print('[ persistent homology in Z%d' % prime)
        kw = {'progress' : True} if self.verbose else {}
        ftime = lambda f,*a,**k: timeit(f, *a, **k) if self.verbose else f(*a, **k)
        self.H['homology'] = ftime(dio.homology_persistence, self.F, prime, **kw)
        self.D['homology'] = dio.init_diagrams(self.H['homology'], self.F)
        return self.D['homology']
    def cohomology(self, prime=2):
        ftime = vtime(self.verbose, '[ persistent cohomology in Z%d' % prime)
        self.H['cohomology'] = ftime(dio.cohomology_persistence, self.F, prime, True)
        self.D['cohomology'] = dio.init_diagrams(self.H['cohomology'], self.F)
        return self.D['cohomology']
    def get_cycle(self, pt):
        H = self.H['homology']
        pair = H.pair(pt.data)
        return H[pair] if pair != H.unpaired else None
    def get_cocycle(self, pt):
        return self.H['cohomology'].cocycle(pt.data)

def np_dgm(D):
    return [np.array([[p.birth, p.death] for p in d]) for d in D]

''''''''''''''''''''''''''''''''
''' SPARSE RIPS PERSISTENCE  '''
class SparseRipsPersistence(SparseRips, Persistence):
    def __init__(self, data, eps=0.1, t=np.inf, dim=2, verbose=True):
        if verbose: print('[ building sparse rips filtration')
        SparseRips.__init__(self, data, eps, t, dim, verbose)
        Persistence.__init__(self, self.incremental_vr(), verbose)

''''''''''''''''''''''''
''' RIPS PERSISTENCE '''
class RipsPersistence(Persistence):
    def __init__(self, data, t=np.inf, dim=2, verbose=True):
        self.data, self.t, self.dim = data, t, dim
        ftime = vtime(verbose, '[ building rips filtration' )
        Persistence.__init__(self, ftime(dio.fill_rips, data, dim, t), verbose)
