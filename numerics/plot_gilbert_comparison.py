"""Plot the recorded Rust Gilbert results against analytical/PPT distances."""
from pathlib import Path
import csv
import os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/minhsd-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rank2_validation import ASTAR, SQRT2, distance2

HERE = Path(__file__).resolve().parent
GFX = HERE.parent / 'gfx'


def main():
    refined = [r for r in csv.DictReader((HERE/'gilbert-refinement.csv').open())
               if r['seed']=='42']
    ppt = list(csv.DictReader((HERE/'rank2-validation.csv').open()))
    assert len(refined)==len(ppt)==75
    assert np.allclose([float(r['alpha']) for r in refined],
                       [float(r['alpha']) for r in ppt],rtol=0,atol=1e-14)
    aa = np.linspace(0,1/SQRT2,1001)
    dd = np.array([distance2(a) for a in aa])
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax = plt.subplots(figsize=(3.6,2.8),layout='constrained')
    ax.plot(aa,dd,color='#236a9b',label='Analytical')
    ax.scatter([float(r['alpha']) for r in ppt[::3]],
               [float(r['sdp_distance2']) for r in ppt[::3]],s=22,
               facecolors='none',edgecolors='#ac3b43',label='PPT projection')
    ax.scatter([float(r['alpha']) for r in refined[::3]],
               [float(r['gilbert_distance2']) for r in refined[::3]],
               s=15,marker='x',color='#277649',
               label='Gilbert')
    ax.set(xlabel=r'$\alpha$',ylabel=r'$D^2_{\mathrm{HS}}$',
           xlim=(-.01,.72),ylim=(-.006,.35))
    ax.legend(frameon=False,fontsize=7.5,loc='upper left')
    ax.axvline(ASTAR,color='.6',lw=.7,ls=':')
    fig.savefig(GFX/'hsd-schmidt-gilbert.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.2,3.6),layout='constrained')
    representative=[.01,.05,.2,.3,ASTAR,.4,.5,.6,1/SQRT2]
    trajectories=[min(refined,key=lambda r:abs(float(r['alpha'])-a))
                  for a in representative]
    for r in trajectories:
        path=HERE/'gilbert-runs'/r['raw_directory']/'input_distances.csv'
        if not path.exists():path=Path(str(path)+'.gz')
        history=np.loadtxt(path,
                           delimiter=',',skiprows=1,ndmin=2)[:,1]
        excess=history-float(r['analytic_distance2'])
        k=np.arange(len(history));valid=(k>0)&(excess>0)
        label=(r'$\alpha_*$' if abs(float(r['alpha'])-ASTAR)<1e-12
               else r'$1/\sqrt{2}$' if abs(float(r['alpha'])-1/SQRT2)<1e-12
               else r'$\alpha='+f'{float(r["alpha"]):.2f}'+r'$')
        ax.loglog(k[valid],excess[valid],lw=1,label=label)
    ax.set(xlabel='Accepted corrections',ylabel=r'$D_{\mathrm{G}}^2-D_{\min}^2$')
    ax.legend(frameon=False,fontsize=7,ncol=2)
    fig.savefig(GFX/'gilbert-convergence.pdf');plt.close(fig)
    print('Saved analytical/PPT/Gilbert comparison and convergence figures.')


if __name__=='__main__':main()
