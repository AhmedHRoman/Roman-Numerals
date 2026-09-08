"""Finite-sample demonstration figure.
(a) Fixed-block plug-in B_O^max compared with the long-time
    bound, vs total observation time, with benchmarks (per-edge observed sum,
    best single-current scalar TUR) and the exact sigma.
(b) Robustness to the block length tau at fixed total time.
Network: 4-state two-cycle; observed set O = cotree (2 of 5 edges), which is
cycle-observing, so the covariance certifies the full hidden-cycle energy."""
import numpy as np
import matplotlib.pyplot as plt

edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 0)]
N, NE = 4, len(edges)
fwd = np.array([2.0, 1.3, 1.7, 1.1, 0.9])
bwd = np.array([1.0, 0.7, 1.0, 0.6, 1.4])
O = [2, 4]                                   # cotree (cycle-observing)

def stationary(W):
    A = np.vstack([W.T, np.ones(N)]); b = np.zeros(N + 1); b[-1] = 1.0
    return np.linalg.lstsq(A, b, rcond=None)[0]

def exact():
    W = np.zeros((N, N))
    for k, (i, j) in enumerate(edges):
        W[i, j] += fwd[k]; W[j, i] += bwd[k]
    for i in range(N): W[i, i] = -W[i].sum()
    p = stationary(W)
    qf = np.array([p[i]*fwd[k] for k,(i,j) in enumerate(edges)])
    qb = np.array([p[j]*bwd[k] for k,(i,j) in enumerate(edges)])
    K, T = qf-qb, qf+qb
    sig_e = K*np.log((T+K)/(T-K)); R = sig_e - 2*K**2/T
    B = np.zeros((N, NE))
    for k,(i,j) in enumerate(edges): B[i,k]=-1; B[j,k]=+1
    _, s, vt = np.linalg.svd(B); C = vt[(s>1e-10).sum():].T
    G = np.zeros((NE, N))
    for k,(i,j) in enumerate(edges): G[k,i]=fwd[k]; G[k,j]=-bwd[k]
    ones = np.ones((N,1))/np.sqrt(N)
    Rb = np.linalg.qr(np.eye(N)-ones@ones.T)[0][:, :N-1]; Gt = G@Rb
    Dinv = np.diag(1.0/T)
    M = Dinv - Dinv@Gt@np.linalg.pinv(Gt.T@Dinv@Gt)@Gt.T@Dinv
    SigK = C@np.linalg.inv(C.T@M@C)@C.T
    return dict(p=p, K=K, T=T, R=R, sig_e=sig_e, sigma=float(sig_e.sum()), SigK=SigK)

E = exact()
K, T, R, sig_e, sigma, SigK = E['K'], E['T'], E['R'], E['sig_e'], E['sigma'], E['SigK']
SigO = SigK[np.ix_(O, O)]
exact_cov  = 2*K[O]@np.linalg.pinv(SigO)@K[O]
exact_hyb  = exact_cov + float(R[O].sum())
exact_Bmax = max(float(sig_e[O].sum()), exact_hyb)
per_edge   = float(sig_e[O].sum())
scalar_tur = 2*max(K[O[i]]**2 / SigO[i, i] for i in range(len(O)))   # best single current

# ---- Gillespie producing per-block edge currents + traffic ----
def sim(tau, nblocks, seed):
    g = np.random.default_rng(seed)
    out = [[] for _ in range(N)]
    for k,(i,j) in enumerate(edges):
        out[i].append((j,k,+1)); out[j].append((i,k,-1))
    rates=[np.array([fwd[k] if d==1 else bwd[k] for (_,k,d) in out[s]]) for s in range(N)]
    tot=np.array([r.sum() for r in rates])
    s=int(g.choice(N, p=E['p']))
    Kb=np.zeros((nblocks,NE)); cnt=np.zeros(NE)
    for b in range(nblocks):
        nf=np.zeros(NE); nb=np.zeros(NE); t=0.0
        while True:
            dt=g.exponential(1/tot[s])
            if t+dt>tau: break
            t+=dt
            dest,k,d=out[s][g.choice(len(out[s]),p=rates[s]/tot[s])]
            if d==1: nf[k]+=1
            else: nb[k]+=1
            s=dest
        Kb[b]=(nf-nb)/tau; cnt+=nf+nb
    return Kb, cnt/(nblocks*tau)

def estimate(Kb, Temp, rcond=1e-3):
    Khat = Kb.mean(0)[O]
    Shat = np.cov(Kb[:, O].T) * tau_global
    Th = Temp[O]
    if np.any(Th <= np.abs(Khat)):
        raise ValueError("Both jump directions must be sampled on each observed edge.")
    cov = 2*Khat@np.linalg.pinv(Shat, rcond=rcond)@Khat
    sg = Khat*np.log((Th+Khat)/(Th-Khat))
    Rh = sg - 2*Khat**2/Th
    hyb = cov + float(Rh.sum())
    return max(float(sg.sum()), hyb), cov, float(sg.sum())

# ===== Panel (a): fixed-block estimates vs total time =====
tau_global = 40.0
Ttot_list = [800, 1600, 3200, 6400, 12800, 25600]
M = 28
a_mean, a_err = [], []
for Ttot in Ttot_list:
    nb = int(Ttot/tau_global)
    vals = [estimate(*sim(tau_global, nb, seed=1000+Ttot*10+m))[0] for m in range(M)]
    a_mean.append(np.mean(vals)); a_err.append(np.std(vals))
    print(f"T={Ttot:6d}  B^max={np.mean(vals):.4f} +/- {np.std(vals):.4f}")

# ===== Panel (b): robustness to block length tau (fixed total time) =====
Ttot_b = 9600
tau_list = [8, 16, 32, 64, 128]
b_mean, b_err = [], []
for tau in tau_list:
    tau_global = tau
    nb = int(Ttot_b/tau)
    vals = [estimate(*sim(tau, nb, seed=7000+int(tau)*7+m))[0] for m in range(M)]
    b_mean.append(np.mean(vals)); b_err.append(np.std(vals))
    print(f"tau={tau:4d}  B^max={np.mean(vals):.4f} +/- {np.std(vals):.4f}")
tau_global = 40.0

# ===== plot =====
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.9))

Tt = np.array(Ttot_list)
axA.axhline(sigma, color='k', ls=':', lw=1.1, label=r'exact $\sigma$')
axA.axhline(exact_Bmax, color='C3', ls='--', lw=1.1,
            label=r'exact $\mathcal{B}_{\mathcal{O}}^{\max}$')
axA.errorbar(Tt, a_mean, yerr=a_err, fmt='D-', color='C3', ms=4, lw=1.3,
             capsize=2, label=r'finite-sample $\widehat{\mathcal{B}}_{\mathcal{O}}^{\max}$')
axA.axhline(per_edge, color='C2', ls='-.', lw=1.1, label='observed-edge sum')
axA.axhline(scalar_tur, color='C0', ls=(0,(1,1)), lw=1.1, label='best single-edge TUR')
axA.set_xscale('log')
axA.set_xlabel('total observation time $t$')
axA.set_ylabel('dissipation bound')
axA.set_ylim(0, max(1.22*sigma, 1.05*np.max(np.array(a_mean)+a_err)))
axA.set_title(r'(a) fixed blocks, $|\mathcal{O}|=2$ of $5$', fontsize=9.5)
axA.legend(fontsize=6.4, frameon=True, facecolor='white', edgecolor='none',
           framealpha=1, loc='lower right')

tl = np.array(tau_list)
axB.axhline(sigma, color='k', ls=':', lw=1.1)
axB.axhline(exact_Bmax, color='C3', ls='--', lw=1.1)
axB.errorbar(tl, b_mean, yerr=b_err, fmt='o-', color='C3', ms=4, lw=1.3, capsize=2)
axB.axhline(per_edge, color='C2', ls='-.', lw=1.1)
axB.set_xscale('log', base=2)
axB.set_xlabel(r'block length $\tau$  (fixed $t=9600$)')
axB.set_ylabel('dissipation bound')
axB.set_ylim(0, max(1.22*sigma, 1.05*np.max(np.array(b_mean)+b_err)))
axB.set_title(r'(b) block-length sensitivity', fontsize=9.5)

plt.tight_layout()
plt.savefig('fig_finite_sample.pdf', bbox_inches='tight')
plt.savefig('fig_finite_sample.png', dpi=160, bbox_inches='tight')
np.savez('fig_finite_sample_data.npz', edges=edges, fwd=fwd, bwd=bwd, observed=O,
         K=K, T=T, Sigma_K=SigK, sigma=sigma, exact_Bmax=exact_Bmax,
         total_times=Ttot_list, block_length=40.0, replicates=M,
         panel_a_mean=a_mean, panel_a_std=a_err, block_lengths=tau_list,
         panel_b_total_time=Ttot_b, panel_b_mean=b_mean, panel_b_std=b_err,
         rcond=1e-3, panel_a_seed_base=1000, panel_b_seed_base=7000)
print(f"\nexact: sigma={sigma:.4f}, Bmax={exact_Bmax:.4f} ({100*exact_Bmax/sigma:.1f}%), "
      f"per_edge={per_edge:.4f}, scalar_tur={scalar_tur:.4f}")
print("saved fig_finite_sample.{pdf,png}")
