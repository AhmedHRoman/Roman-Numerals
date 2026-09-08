"""Kinesin demonstration figure (Liepelt-Lipowsky 6-state model).
(a) Exact certified dissipation bound vs load force: the displacement-only
    certificate collapses at the stall force (futile hydrolysis, no net step),
    while step + one chemical transition certifies the full cycle energy.
(b) Finite-sample validation at the stall force from simulated trajectories.
"""
import numpy as np
import matplotlib.pyplot as plt
import kinesin_step1 as ks
import hashlib
import inspect

ATP, ADP, Pi = 1000.0, 1.0, 1.0
F_STALL = ks.ll_stall_force(ATP, ADP, Pi)

# ---------------- panel (a): exact certified bound vs force ----------------
Fgrid = np.linspace(0, 10, 41)
disp_only, step_chem, full_E, sig = [], [], [], []
for F in Fgrid:
    N, edges, wf, wb, mech, chem = ks.ll_edge_rates(ATP, ADP, Pi, F)
    net = ks.MarkovNet(N, edges, wf, wb)
    full = 2*float(net.K @ net.M @ net.K)
    disp_only.append(net.covariance_bound([mech]))
    step_chem.append(net.covariance_bound([mech, chem[1]]))
    full_E.append(full); sig.append(net.sigma)
disp_only = np.array(disp_only); step_chem = np.array(step_chem)
full_E = np.array(full_E); sig = np.array(sig)

# ---------------- panel (b): finite-sample at stall ----------------
N, edges, wf, wb, mech, chem = ks.ll_edge_rates(ATP, ADP, Pi, F_STALL)
net = ks.MarkovNet(N, edges, wf, wb)
NE = len(edges)
O = [mech, chem[1]]                              # step + one chemical
exact_sc = net.covariance_bound(O)
exact_do = net.covariance_bound([mech])

def gillespie_blocks(wf, wb, tau, nblocks, seed):
    # precompute per-state outgoing (dest, edge, dir, rate), cumulative for fast sampling
    g = np.random.default_rng(seed)
    dest_s, edge_s, dir_s, cum_s, tot_s = [], [], [], [], []
    for s in range(net.N):
        ds, es, di, rs = [], [], [], []
        for k,(i,j) in enumerate(edges):
            if i == s: ds.append(j); es.append(k); di.append(+1); rs.append(wf[k])
            if j == s: ds.append(i); es.append(k); di.append(-1); rs.append(wb[k])
        rs = np.array(rs)
        dest_s.append(np.array(ds)); edge_s.append(np.array(es))
        dir_s.append(np.array(di)); cum_s.append(np.cumsum(rs)); tot_s.append(rs.sum())
    tot_s = np.array(tot_s)
    s = int(g.choice(net.N, p=net.p))
    Kb = np.zeros((nblocks, NE))
    for b in range(nblocks):
        net_k = np.zeros(NE); t = 0.0
        while True:
            t += -np.log(g.random())/tot_s[s]
            if t > tau: break
            u = g.random()*tot_s[s]; c = np.searchsorted(cum_s[s], u)
            net_k[edge_s[s][c]] += dir_s[s][c]
            s = int(dest_s[s][c])
        Kb[b] = net_k / tau
    return Kb

RCOND = 1e-3
SEED_BASE = 100

def estimate(Kb, obs, rcond=RCOND):
    Khat=Kb.mean(0)[obs]; Shat=np.cov(Kb[:,obs].T)*TAU
    Shat=np.atleast_2d(Shat)
    return 2*Khat@np.linalg.pinv(Shat,rcond=rcond)@Khat

import os
TAU=50.0
Tlist=[1000,2000,4000,8000]
M=14
CACHE='fig_kinesin_sim_stall_exact.npz'
provenance = dict(wf=wf, wb=wb, edges=np.asarray(edges), observed=np.asarray(O),
                  initial_p=net.p, rcond=RCOND, seed_base=SEED_BASE,
                  simulator_sha256=hashlib.sha256(
                      (inspect.getsource(gillespie_blocks) +
                       inspect.getsource(estimate)).encode()).hexdigest())
if os.path.exists(CACHE):
    d=np.load(CACHE)
    same_grid = np.array_equal(d['Tlist'], np.array(Tlist))
    same_params = (abs(float(d['F_STALL']) - F_STALL) < 1e-9 and
                   abs(float(d['TAU']) - TAU) < 1e-12 and
                   int(d['M']) == M)
    same_inputs = all(key in d and np.array_equal(d[key], np.asarray(value))
                      for key, value in provenance.items())
    if same_grid and same_params and same_inputs:
        sc_mean,sc_err,do_mean=list(d['sc_mean']),list(d['sc_err']),list(d['do_mean'])
        print('loaded cached simulation', CACHE)
    else:
        sc_mean,sc_err,do_mean=[],[],[]
else:
    sc_mean,sc_err,do_mean=[],[],[]
if len(sc_mean) == 0:
    for Ttot in Tlist:
        nb=int(Ttot/TAU)
        sc=[]; do=[]
        for m in range(M):
            Kb=gillespie_blocks(wf,wb,TAU,nb,seed=SEED_BASE+Ttot+m)
            sc.append(estimate(Kb,O)); do.append(estimate(Kb,[mech]))
        sc_mean.append(np.mean(sc)); sc_err.append(np.std(sc)); do_mean.append(np.mean(do))
        print(f"T={Ttot:6d}  step+chem={np.mean(sc):7.2f}+/-{np.std(sc):5.2f}  "
              f"disp-only={np.mean(do):6.3f}  (exact sc={exact_sc:.2f})")
    np.savez(CACHE, sc_mean=sc_mean, sc_err=sc_err, do_mean=do_mean,
             Tlist=np.array(Tlist), F_STALL=F_STALL, TAU=TAU, M=M, **provenance)

# ---------------- plot: single column-width panel with inset ----------------
fig, ax = plt.subplots(figsize=(3.4, 2.9))

ax.plot(Fgrid, step_chem, '-', color='C0', lw=2.0, label='step + 1 chemical')
ax.plot(Fgrid, disp_only, '-', color='C3', lw=2.0, label='displacement only')
ax.plot(Fgrid, full_E, ':', color='gray', lw=1.2, label=r'full $2K^\top M_T K$')
ax.axvline(F_STALL, ymax=0.40, color='k', ls='--', lw=0.8, alpha=0.6)
ax.text(F_STALL-0.2, 0.34*step_chem.max(), 'stall', fontsize=7, rotation=90,
        va='center', ha='right', color='k', alpha=0.7)
ax.set_xlabel('load force $F$ (pN)')
ax.set_ylabel(r'certified energy $2K_O^\top\Sigma_O^+K_O$  ($k_BT$/s)')
ax.set_xlim(0,10); ax.set_ylim(0, 1.05*step_chem.max())
ax.legend(fontsize=7, frameon=False, loc='lower left',
          bbox_to_anchor=(0.02, 0.02), handlelength=1.7, borderaxespad=0)

# inset: fixed-block finite-sample estimates at stall
axin = ax.inset_axes([0.57, 0.54, 0.40, 0.32])
axin.set_facecolor('white')
axin.patch.set_alpha(1.0)
Tt=np.array(Tlist)
axin.axhline(exact_sc, color='C0', ls='--', lw=1.0)
axin.errorbar(Tt, sc_mean, yerr=sc_err, fmt='o-', color='C0', ms=3, lw=1.0, capsize=1.5)
axin.plot(Tt, do_mean, 's-', color='C3', ms=3, lw=1.0)
axin.set_xscale('log'); axin.set_xticks(Tlist)
axin.set_xticklabels([f'{t//1000}k' for t in Tlist], fontsize=5.5)
axin.minorticks_off()
axin.tick_params(axis='y', labelsize=5.5)
sc_lower = np.asarray(sc_mean) - np.asarray(sc_err)
sc_upper = np.asarray(sc_mean) + np.asarray(sc_err)
axin.set_ylim(min(-2, float(sc_lower.min()) - 2),
              1.08 * max(exact_sc, float(sc_upper.max()), max(do_mean)))
axin.set_xlabel('$t$ (s)', fontsize=6, labelpad=1)
axin.set_title('finite sample at stall', fontsize=6.2, pad=2)

plt.tight_layout()
plt.savefig('fig_kinesin.pdf', bbox_inches='tight')
plt.savefig('fig_kinesin.png', dpi=200, bbox_inches='tight')
np.savez('fig_kinesin_data.npz', force_pN=Fgrid, displacement_bound=disp_only,
         step_chemical_bound=step_chem, full_quadratic_cost=full_E, sigma=sig,
         stall_pN=F_STALL, total_seconds=Tt, block_seconds=TAU, replicates=M,
         step_chemical_mean=sc_mean, step_chemical_std=sc_err,
         displacement_mean=do_mean, exact_stall_bound=exact_sc,
         ATP_uM=ATP, ADP_uM=ADP, phosphate_uM=Pi, **provenance)
print(f"\nexact: step+chem={exact_sc:.2f}, disp-only={exact_do:.3f}, "
      f"sigma(stall)={net.sigma:.1f}")
print(f"stall force = {F_STALL:.4f} pN")
print("saved fig_kinesin.{pdf,png}")
