from matplotlib.pyplot import Polygon
import matplotlib.pyplot as plt
import numpy as np

plt.ion()
fig, ax = plt.subplots(1, 2, figsize=(9,4), sharex=True, sharey=True)

def plot_simplices(axis, R, F):
    for s in F:
        x = R.data[list(s)]
        if s.dimension() == 0:
            axis.plot(x[:,0], x[:,1], 'o', c='black', zorder=2)
        elif s.dimension() == 1:
            axis.plot(x[:,0], x[:,1], c='red', alpha=0.7, zorder=1)
        elif s.dimension() == 2:
            axis.add_patch(Polygon(x, color='black', alpha=0.1, zorder=0))

def plot_dgm(axis, D, **kw):
    if 's' in kw:
        kw['markersize'] = kw['s']
        del kw['s']
    axis.plot([0,1], [0,1], c='black', alpha=0.5, zorder=0)
    for d in D:
        dgm = np.array([[p.birth, p.death] for p in d])
        if len(dgm): axis.plot(dgm[:,0], dgm[:,1], 'o', **kw)
