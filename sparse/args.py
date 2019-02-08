from argparse import ArgumentParser
from . import *

''''''''''''''''''
'''  DEFAULTS  '''
''''''''''''''''''
DEF = {'epsilon' : 0.1, 'noise' : 0.2, 'dim' : 2, 'prime' : 2,
        'n' : 100, 'uniform' : False, 'function' : 'double',
        'thresh' : 2 * np.sqrt(2), 'verbose' : False,
        'cohomology' : False}

''''''''''''''''''
''' FUNCTIONS  '''
''''''''''''''''''
FUN = {'circle' : circle, 'double' : double_circle}

''''''''''''''''''
''' PARSER DEF '''
''''''''''''''''''
parser = ArgumentParser(description='sparse rips (co)homology')
parser.add_argument('--epsilon', '-e', type=float, default=DEF['epsilon'],
                        help='sparsity. default: %0.2f' % DEF['epsilon'])
parser.add_argument('--noise', '-o', type=float, default=DEF['noise'],
                        help='noise multiplier. default: %0.1f' % DEF['noise'])
parser.add_argument('--dim', '-d', type=int, default=DEF['dim'],
                        help='max rips dimension. default: %d' % DEF['dim'])
parser.add_argument('--prime', '-p', type=int, default=DEF['prime'],
                        help='prime (co)homology coefficient. default: %d' % DEF['prime'])
parser.add_argument('--function', '-f', choices=FUN.keys(), default=DEF['function'],
                        help='shape. default: %s' % DEF['function'])
parser.add_argument('--thresh', '-t', type=float, default=DEF['thresh'],
                        help='rips cutoff. default: %0.3f' % DEF['thresh'])
parser.add_argument('--uniform', '-u', action='store_true',
                        help='uniformly sample shape. default: %s' % str(DEF['uniform']))
parser.add_argument('--n', '-n', type=int, default=DEF['n'],
                        help='number of points. default: %d' % DEF['n'])
parser.add_argument('--verbose', '-v', action='store_true',
                        help='verbose output. default: %d' % DEF['verbose'])
parser.add_argument('--cohomology', '-c', action='store_true',
                        help='persistent cohomology. default: persistent homology')
