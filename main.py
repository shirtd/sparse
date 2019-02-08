#!/usr/bin/env python
from sparse.args import *
from sparse import *

def test_sparse(data, args):
    t = (1. + args.epsilon) * args.thresh
    if args.verbose:
        start_time = time.time()
    R = SparseRipsPersistence(data, args.epsilon, t, args.dim, args.verbose)
    D = (R.cohomology if args.cohomology else R.homology)(args.prime)
    if args.verbose:
        print('TOTAL: %s seconds' % (time.time() - start_time))
    return R, D

def test_rips(data, args):
    if args.verbose:
        start_time = time.time()
    R = RipsPersistence(data, args.thresh, args.dim, args.verbose)
    D = (R.cohomology if args.cohomology else R.homology)(args.prime)
    if args.verbose:
        print('TOTAL: %s seconds' % (time.time() - start_time))
    return R, D

if __name__ == '__main__':
    args = parser.parse_args()
    data = FUN[args.function](args.n, args.uniform, args.noise)

    R, D = test_sparse(data, args)
    _R, _D = test_rips(data, args)

    if args.plot:
        from sparse.plot import *
        plot_dgm(ax[0], D, s=2)
        plot_dgm(ax[1], _D, s=2)
        plt.pause(0.1)
        sys.stdout.write('[ press any key to exit ] ')
        sys.stdout.flush()
        sys.stdin.readline(1)
