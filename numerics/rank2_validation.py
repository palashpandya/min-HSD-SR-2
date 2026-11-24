"""Reproduce the revised rank-2 manuscript's figures and independent checks.

Run from any directory: python3 rank2_validation.py
Dependencies: numpy, scipy, sympy, matplotlib, cvxpy, clarabel.
The analytical formula and unrestricted density-matrix PPT projection are
implemented independently. No commutation/saturation ansatz is used in the SDP.
"""
from pathlib import Path
import csv
import json
import os
import platform
import warnings
import numpy as np
import sympy as sp
import scipy
import scipy.optimize
import cvxpy as cp
import clarabel
os.environ.setdefault('MPLCONFIGDIR','/tmp/minhsd-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
GFX = HERE.parent / 'gfx'
ASTAR = np.sqrt((3-np.sqrt(5))/6)
SQRT2 = np.sqrt(2.)


def coefficients(alpha):
    beta = np.sqrt(max(0., 1-alpha**2))
    n = alpha*beta
    if alpha >= ASTAR:
        return alpha**2-n/3, beta**2-n/3, 2*n/3
    if alpha == 0:
        return 0., 1., 0.
    # Solve the physical stationary cubic directly, independently of Cardano.
    t = scipy.optimize.brentq(
        lambda t: 6*t**3-9*t**2+(4+2*n)*t-alpha**2-n,
        0., n/2, xtol=5e-16)
    return t*t, (1-t)**2, 2*t*(1-t)


def css(alpha):
    e1,e2,e3 = coefficients(alpha)
    return np.array([[e1,0,0,e3/2], [0,e3/2,0,0],
                     [0,0,e3/2,0], [e3/2,0,0,e2]])


def target(alpha, dim=2):
    v = np.zeros(dim*dim)
    v[0],v[dim+1] = alpha, np.sqrt(max(0.,1-alpha*alpha))
    return np.outer(v,v)


def distance2(alpha):
    return float(np.sum((target(alpha)-css(alpha))**2))


def gradient(alpha):
    beta = np.sqrt(1-alpha*alpha)
    if alpha >= ASTAR:
        return 8*alpha*(1-2*alpha*alpha)/3
    t=np.sqrt(coefficients(alpha)[0])
    return 4*(alpha*(1-2*t)-(1-2*alpha*alpha)*t*(1-t)/beta)


def partial_transpose(x,dim):
    return cp.bmat([[x[(i//dim)*dim+j%dim,(j//dim)*dim+i%dim]
                     for j in range(dim*dim)] for i in range(dim*dim)])


def solve_ppt(alpha,dim=2):
    rho=target(alpha,dim)
    x=cp.Variable((dim*dim,dim*dim),symmetric=True)
    prob=cp.Problem(cp.Minimize(cp.sum_squares(x-rho)),
                    [x >> 0, partial_transpose(x,dim) >> 0, cp.trace(x)==1])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Solution may be inaccurate')
        prob.solve(solver='CLARABEL',tol_gap_abs=1e-11,tol_gap_rel=1e-11,
                   tol_feas=1e-11,max_iter=300)
    assert prob.status in ['optimal','optimal_inaccurate'],prob.status
    mat=np.asarray(x.value)
    pt=mat.reshape(dim,dim,dim,dim).transpose(0,3,2,1).reshape(dim*dim,dim*dim)
    return dict(distance2=float(prob.value),status=prob.status,
                min_eig=float(np.linalg.eigvalsh(mat).min()),
                min_pt_eig=float(np.linalg.eigvalsh(pt).min()),
                trace_error=float(abs(np.trace(mat)-1)))


def algebra_checks():
    a,b,n,t,h,d=sp.symbols('a b n t h d',real=True)
    p=lambda t:6*t**3-9*t**2+(4+2*n)*t-a*a-n
    lower=n*(1-3*n)*(4-n)/4
    assert sp.expand(p(n/2)-lower-(2*n*n-a*a))==0
    F=a*a*b*b+(a*a-t*t)**2+(b*b-(1-t)**2)**2+(a*b-2*t*(1-t))**2
    g=sp.diff(F,t)/4
    assert sp.expand(g.subs(b*b,1-a*a)-p(t).subs(n,a*b))==0
    Fn=F.subs(b,sp.sqrt(1-a*a))
    env=4*(a*(1-2*t)-(1-2*a*a)*t*(1-t)/sp.sqrt(1-a*a))
    assert sp.simplify(sp.diff(Fn,a)-env)==0
    descent=2*h*h-4*d*h+2*(d-h)**2-2*d*d
    assert sp.expand(descent-(-8*d*h+4*h*h))==0
    orig=1+a*a+b*b+(1-a-b)**2
    squares=sp.Rational(4,3)+(a-sp.Rational(1,3))**2+(b-sp.Rational(1,3))**2+(a+b-sp.Rational(2,3))**2
    assert sp.expand(orig-squares)==0
    return ['stationary cubic','physical-root bound identity','envelope derivative',
            'trace-preserving descent','equal-redistribution square completion']


def concurrence_checks():
    c,delta,y,t=sp.symbols('c delta y t',real=True)
    a2,b2=(1-delta)/2,(1+delta)/2
    F=c*c/4+(a2-t*t)**2+(b2-(1-t)**2)**2+(c/2-2*t*(1-t))**2
    Q=sp.Rational(7,8)-c/2-delta*y+(2*c-1)*y*y/4+3*y**4/8
    H=(7-4*c-6*delta*y+(2*c-1)*y*y)/8
    cubic=3*y**3+(2*c-1)*y-2*delta
    assert sp.expand(F.subs(t,(1-y)/2)-Q).subs(delta**2,1-c*c).expand()==0
    assert sp.expand(2*sp.diff(Q,y)-cubic)==0
    assert sp.expand(Q-H-y*cubic/8)==0
    # Verify the endpoint series directly from stationarity and the objective.
    k2,k3=sp.symbols('k2 k3')
    ts=c/8+k2*c*c+k3*c**3
    p=6*t**3-9*t*t+(4+c)*t-(1-sp.sqrt(1-c*c))/2-c/2
    ps=sp.series(p.subs(t,ts),c,0,4).removeO().expand()
    ks=sp.solve([ps.coeff(c,2),ps.coeff(c,3)],[k2,k3])
    series=sp.series(F.subs({delta:sp.sqrt(1-c*c),t:ts.subs(ks)}),c,0,4).removeO()
    assert sp.expand(series-3*c*c/8+15*c**3/128)==0
    max_distance=max_cubic=0.
    for cv in np.linspace(0.,2/3,1001):
        dv=np.sqrt(1-cv*cv)
        g=np.cbrt(9*dv+np.sqrt(8*cv**3-93*cv*cv+6*cv+80))
        yv=(g+(1-2*cv)/g)/3
        hv=(7-4*cv-6*dv*yv+(2*cv-1)*yv*yv)/8
        # Stable inversion of concurrence; compare to the original alpha formula.
        alpha=cv/np.sqrt(2*(1+dv))
        max_distance=max(max_distance,abs(hv-distance2(alpha)))
        max_cubic=max(max_cubic,abs(3*yv**3+(2*cv-1)*yv-2*dv))
    assert max_distance<1e-12 and max_cubic<1e-12
    return dict(samples=1001,max_distance2_error=float(max_distance),
                max_cubic_residual=float(max_cubic))


def formula_checks():
    max_cardano=max_fd=max_decomp=0.
    for a in np.linspace(.0001,ASTAR,401):
        b=np.sqrt(1-a*a);n=a*b
        g=9*(b*b-a*a)+2*np.sqrt(16*n**3-93*n*n+3*n+20)
        e2=(3+(1-4*n)/np.cbrt(g)+np.cbrt(g))**2/36
        e=coefficients(a)
        max_cardano=max(max_cardano,abs(e2-e[1]))
        t=np.sqrt(e[0])
        assert -1e-14<=t<=n/2+1e-14
    for a in np.linspace(.001,1/SQRT2-.001,401):
        if abs(a-ASTAR)<1e-4:continue
        fd=(distance2(a+1e-6)-distance2(a-1e-6))/2e-6
        max_fd=max(max_fd,abs(fd-gradient(a)))
        e1,e2,e3=coefficients(a)
        s=e3/(2*np.sqrt(e1*e2));q=s*(np.sqrt(e1)+np.sqrt(e2))**2
        a2=np.sqrt(e1)/(np.sqrt(e1)+np.sqrt(e2));b2=1-a2
        rebuilt=np.zeros((4,4),complex)
        for k in range(3):
            ph=2*np.pi*k/3
            u=np.array([np.sqrt(a2),np.sqrt(b2)*np.exp(1j*ph)])
            v=np.kron(u,u.conj());rebuilt+=q*np.outer(v,v.conj())/3
        rebuilt[0,0]+=(1-s)*e1;rebuilt[3,3]+=(1-s)*e2
        max_decomp=max(max_decomp,float(abs(rebuilt-css(a)).max()))
        assert min(e1,e2,e3)>0
        assert 0<=q<=1+1e-12 and 0<=s<=1+1e-12
    assert max_cardano<1e-12 and max_fd<1e-7 and max_decomp<1e-12
    return dict(max_cardano_error=max_cardano,max_gradient_fd_error=max_fd,
                max_separable_reconstruction_error=max_decomp,
                concurrence_relation=concurrence_checks())


def main():
    algebra=algebra_checks();formula=formula_checks()
    grid=np.unique(np.r_[np.arange(0.,.701,.01),1/SQRT2,ASTAR,
                         ASTAR+np.array([-1e-5,1e-5])])
    rows=[]
    for a in grid:
        sol=solve_ppt(a);exact=distance2(a)
        rows.append(dict(alpha=float(a),analytic_distance2=exact,
                         sdp_distance2=sol['distance2'],
                         absolute_error=abs(sol['distance2']-exact),
                         min_eig=sol['min_eig'],min_pt_eig=sol['min_pt_eig'],
                         trace_error=sol['trace_error'],status=sol['status']))
    with (HERE/'rank2-validation.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    embeds=[]
    for a in [.05,.2,ASTAR,.5,1/SQRT2]:
        v=solve_ppt(a,3)
        embeds.append(dict(alpha=a,absolute_error=abs(v['distance2']-distance2(a))))
    error=max(x['absolute_error'] for x in rows)
    assert error<1e-8,error
    report=dict(samples=len(rows),max_distance2_error=error,
                max_trace_error=max(x['trace_error'] for x in rows),
                min_eig=min(x['min_eig'] for x in rows),
                min_pt_eig=min(x['min_pt_eig'] for x in rows),
                embedded_checks=embeds,algebra=algebra,formula_checks=formula,
                versions=dict(python=platform.python_version(),
                              clarabel=clarabel.__version__,
                              numpy=np.__version__,scipy=scipy.__version__,
                              sympy=sp.__version__,cvxpy=cp.__version__,
                              matplotlib=matplotlib.__version__))
    (HERE/'rank2-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    aa=np.linspace(0,1,2001);nn=aa*np.sqrt(np.maximum(0,1-aa*aa))
    yy=4*nn*(nn-1/3)
    np.savetxt(HERE/'positivity-curve.csv',np.c_[aa,yy],delimiter=',',
               header='alpha,4e1e2_minus_e3_squared',comments='')
    fig,ax=plt.subplots(figsize=(4.6,3.0),layout='constrained')
    ax.plot(aa,yy,color='#236a9b');ax.axhline(0,color='.5',lw=.7)
    ax.scatter([ASTAR,np.sqrt(1-ASTAR*ASTAR),1/SQRT2],[0,0,1/3],s=18,color='#9c3647',zorder=3)
    ax.set(xlabel=r'$\alpha$',ylabel=r'$4e_1e_2-e_3^2$',xlim=(0,1),ylim=(-.13,.37))
    ax.annotate(r'$\alpha_*$',(ASTAR,0),xytext=(8,-18),textcoords='offset points')
    ax.annotate(r'$1/3$',(1/SQRT2,1/3),xytext=(8,-6),textcoords='offset points')
    fig.savefig(GFX/'eigenvalue-positivity.pdf');plt.close(fig)
    fine=np.linspace(0,1/SQRT2,1001);dd=np.array([distance2(a) for a in fine])
    fig,ax=plt.subplots(figsize=(4.6,3.0),layout='constrained')
    ax.plot(fine,dd,color='#236a9b',label='Analytical')
    ax.scatter(grid[::3],[r['sdp_distance2'] for r in rows[::3]],s=19,
               facecolors='none',edgecolors='#ac3b43',label='Full PPT projection')
    ax.axvline(ASTAR,color='.6',lw=.7,ls=':')
    ax.set(xlabel=r'$\alpha$',ylabel=r'$D_{\mathrm{HS}_{\min}}^2$',
           xlim=(0,1/SQRT2),ylim=(0,.35));ax.legend(frameon=False,fontsize=8)
    fig.savefig(GFX/'hsd-schmidt-gilbert.pdf');plt.close(fig)
    deriv=np.array([gradient(a) for a in fine])
    np.savetxt(HERE/'distance-and-gradient.csv',np.c_[fine,dd,deriv],delimiter=',',
               header='alpha,analytic_distance2,derivative_distance2',comments='')
    fig,ax=plt.subplots(figsize=(4.6,3.0),layout='constrained')
    left=fine<=ASTAR;right=fine>=ASTAR
    ax.plot(fine[left],deriv[left],color='#ac3b43',label='Boundary branch')
    ax.plot(fine[right],deriv[right],color='#236a9b',label='Simple branch')
    ax.axvline(ASTAR,color='.6',lw=.7,ls=':')
    ax.set(xlabel=r'$\alpha$',ylabel=r'$\mathrm{d}D_{\mathrm{HS}_{\min}}^2/\mathrm{d}\alpha$',
           xlim=(0,1/SQRT2),ylim=(0,.78));ax.legend(frameon=False,fontsize=8)
    fig.savefig(GFX/'hsd-gradient.pdf');plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
