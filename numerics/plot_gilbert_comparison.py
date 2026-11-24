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
    results = list(csv.DictReader((HERE/'gilbert-comparison.csv').open()))
    baseline = [r for r in results if r['seed']=='42' and r['correction_budget']=='200']
    longer = [r for r in results if r['seed']=='42' and r['correction_budget']=='2000']
    refined_file=HERE/'gilbert-refinement.csv'
    refined=([r for r in csv.DictReader(refined_file.open()) if r['seed']=='42']
             if refined_file.exists() else [])
    ppt = list(csv.DictReader((HERE/'rank2-validation.csv').open()))
    assert len(baseline)==len(ppt)==75 and len(longer)==9
    aa = np.linspace(0,1/SQRT2,1001)
    dd = np.array([distance2(a) for a in aa])
    x = np.array([float(r['alpha']) for r in baseline])
    y = np.array([float(r['gilbert_distance2']) for r in baseline])
    gap = np.array([float(r['distance2_excess']) for r in baseline])
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig,(ax,err) = plt.subplots(2,1,figsize=(3.6,3.8),sharex=True,
                                gridspec_kw={'height_ratios':[2,1]},
                                layout='constrained')
    ax.plot(aa,dd,color='#236a9b',label='Analytical')
    ax.scatter([float(r['alpha']) for r in ppt[::3]],
               [float(r['sdp_distance2']) for r in ppt[::3]],s=22,
               facecolors='none',edgecolors='#ac3b43',label='PPT projection')
    displayed=refined if refined else baseline
    ax.scatter([float(r['alpha']) for r in displayed[::3]],
               [float(r['gilbert_distance2']) for r in displayed[::3]],
               s=15,marker='x',color='#277649',
               label='Gilbert (refined)' if refined else 'Gilbert (budget 200)')
    ax.set(ylabel=r'$D^2_{\mathrm{HS}}$',xlim=(-.01,.72),ylim=(-.006,.35))
    ax.legend(frameon=False,fontsize=7.5,loc='upper left')
    valid=gap>0
    err.plot(x[valid],gap[valid],color='#277649',lw=1,
             label='Initial settings' if refined else 'Budget 200')
    comparison=refined if refined else longer
    comparison=[r for r in comparison if float(r['distance2_excess'])>0]
    err.scatter([float(r['alpha']) for r in comparison],
                [float(r['distance2_excess']) for r in comparison],marker='D',s=10,
                color='#9f662d',label='Refined settings' if refined else 'Budget 2,000',zorder=3)
    err.set(yscale='log',xlabel=r'$\alpha$',
            ylabel=r'$D_{\mathrm{G}}^2-D_{\min}^2$')
    err.legend(frameon=False,fontsize=7.5,loc='upper right')
    for axis in [ax,err]:axis.axvline(ASTAR,color='.6',lw=.7,ls=':')
    fig.savefig(GFX/'hsd-schmidt-gilbert.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.2,3.6),layout='constrained')
    trajectories=longer
    if refined:
        trajectories=[min(refined,key=lambda r:abs(float(r['alpha'])-float(old['alpha'])))
                      for old in longer]
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
