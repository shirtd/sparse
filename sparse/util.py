from multiprocessing import Pool
from numpy.random import normal
from functools import partial
from tqdm import tqdm, trange
import numpy.linalg as la
import pickle as pkl
import numpy as np
import sys, os
import time

'''''''''''''''
'''' DATA '''''
'''''''''''''''

def circle(n=20, uniform=False, noise=0.1, r=1.):
    t = np.linspace(0, 1, n, False) if uniform else np.random.rand(n)
    e = r * (1 + noise * (2 * np.random.rand(n) - 1)) if noise else r
    return np.array([e*np.cos(2 * np.pi * t),
                    e*np.sin(2*np.pi*t)]).T.astype(np.float32)

def double_circle(n=50, uniform=False, noise=0.1, r=(1., 0.7)):
    p1 = circle(int(n * r[0] / sum(r)), uniform, noise, r[0])
    p2 = circle(int(n * r[1] / sum(r)), uniform, noise, r[1])
    return np.vstack([p1 - np.array([r[0] + noise, 0.]),
                    p2 + np.array([r[1] + noise, 0.])])

def torus(n=1000, R=0.7, r=0.25):
    t = 2*np.pi * np.random.rand(2, n)
    x = (R + r * np.cos(t[1])) * np.cos(t[0])
    y = (R + r * np.cos(t[1])) * np.sin(t[0])
    z = r * np.sin(t[1])
    return np.vstack([x, y, z]).T


''''''''''''''
''' I/O UTIL ''
'''''''''''''''

def delete_line():
    sys.stdout.write('\x1b[1A')
    sys.stdout.write('\x1b[2K')

def wait(s=''):
    ret = raw_input(s)
    delete_line()
    return ret

def save_pkl(fname, x):
    with open(fname, 'w') as f:
        pkl.dump(x, f)
    return fname

def load_pkl(fname):
    with open(fname, 'r') as f:
        x = pkl.load(f)
    return x

def save_state(fname, data, **kwargs):
    dout = {'data' : data, 'args' : kwargs}
    print('[ saving %s' % fname)
    save_pkl(fname, dout)
    return dout

def query_save(data, fpath='.', **kwargs):
    ret = raw_input('[ save as: %s/' % fpath)
    if not ret: return None, None
    flist = os.path.split(ret)
    for dir in flist[:-1]:
        fpath = os.path.join(fpath, dir)
        if not os.path.isdir(fpath):
            print(' | creating directory %s' % fpath)
            os.mkdir(fpath)
    fname = os.path.join(fpath, flist[-1])
    return fname, save_state(fname, data, **kwargs)

def load_args(*keys):
    if len(sys.argv) > 1:
        try:
            x = load_pkl(sys.argv[1])
            return [x[k] for k in keys]
        except e:
            print(e)
    return []

def timeit(f, *args, **kw):
    start_time = time.time()
    x = f(*args, **kw)
    t = (time.time() - start_time)
    print(' --- %s seconds ---' % t)
    return x

def vtimeit(desc, f, *args, **kw):
    sys.stdout.write(desc)
    sys.stdout.flush()
    start_time = time.time()
    x = f(*args, **kw)
    t = (time.time() - start_time)
    print(' : %s seconds' % t)
    return x

def vrange(verbose, desc):
    return lambda *args: trange(*args, desc=desc) if verbose else range(*args)

def viter(verbose, desc):
    return lambda arg: tqdm(arg, desc=desc) if verbose else arg

def vtime(verbose, desc):
    return lambda f, *args, **kw: vtimeit(desc, f, *args, **kw) if verbose else f(*args, **kw)


'''''''''''''''''
'' THREAD UTIL ''
'''''''''''''''''


def tmap(fun, n, frange):
    pool = Pool()
    try:
        y = pool.map(fun, frange(n))
    except KeyboardInterrupt as e:
        print(e)
        pool.close()
        pool.join()
        sys.exit()
    pool.close()
    pool.join()
    return y

def pmap(fun, x, *args):
    pool = Pool()
    f = partial(fun, *args)
    try:
        # y = pool.map(f, tqdm(x))
        y = pool.map(f, x)
    except KeyboardInterrupt as e:
        print(e)
        pool.close()
        pool.join()
        sys.exit()
    pool.close()
    pool.join()
    return y

def cmap(funct, x, *args):
    fun = partial(funct, *args)
    f = partial(map, fun)
    pool = Pool()
    try:
        # y = pool.map(f, tqdm(x))
        y = pool.map(f, x)
    except KeyboardInterrupt as e:
        print(e)
        pool.close()
        pool.join()
        sys.exit()
    pool.close()
    pool.join()
    return y
