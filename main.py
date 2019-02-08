#!/usr/bin/env python
from sparse.plot import *
from sparse.args import *
from sparse import *

if __name__ == '__main__':
    args = parser.parse_args()
    ftime = lambda f,*a,**k: timeit(f,*a,**k) if args.verbose else f(*a,**k)
    data = FUN[args.function](args.n, args.uniform, args.noise)

    t = (1. + args.epsilon) * args.thresh
    R = ftime(SparseRipsPersistence, data, args.epsilon, t, args.dim, args.verbose)
    D = (R.cohomology if args.cohomology else R.homology)(args.prime)
    plot_dgm(ax[0], D, s=2)

    _R = ftime(RipsPersistence, data, args.thresh, args.dim, args.verbose)
    _D = (_R.cohomology if args.cohomology else _R.homology)(args.prime)
    plot_dgm(ax[1], _D, s=2)

    raw_input('[ press any key to exit ] ')
