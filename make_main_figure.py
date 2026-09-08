"""Two-panel figure spanning both PRL columns:
  Left  (A): 3x3 torus, unscreened physical current, four estimators.
  Right (B): nonuniform 4-state two-cycle, gamma_G sweep over five values.
All exact-theory; no Gillespie noise. For the torus we enumerate enough
random subsets per size that the subset spread is tight."""
import numpy as np, itertools, matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def draw_network(ax, pos, edges, node_size=40, lw=0.9, alpha=0.85):
    for (i, j) in edges:
        ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],
                'k-', lw=lw, alpha=alpha, zorder=1)
    ax.scatter(pos[:,0], pos[:,1], s=node_size, c='C0', edgecolor='k',
               zorder=2, linewidths=0.6)
    ax.set_aspect('equal'); ax.axis('off')
    pad = 0.25
    ax.set_xlim(pos[:,0].min()-pad, pos[:,0].max()+pad)
    ax.set_ylim(pos[:,1].min()-pad, pos[:,1].max()+pad)

rng = np.random.default_rng(0)

def stationary(W):
    N = W.shape[0]
    A = np.vstack([W.T, np.ones(N)]); b = np.zeros(N+1); b[-1] = 1.0
    return np.linalg.lstsq(A, b, rcond=None)[0]

def network_objects(N, edges, fwd, bwd, _store_rates=True):
    NE = len(edges)
    W = np.zeros((N,N))
    for k,(i,j) in enumerate(edges):
        W[i,j] += fwd[k]; W[j,i] += bwd[k]
    for i in range(N): W[i,i] = -W[i,:].sum()
    p = stationary(W)
    qf = np.array([p[i]*fwd[k] for k,(i,j) in enumerate(edges)])
    qb = np.array([p[j]*bwd[k] for k,(i,j) in enumerate(edges)])
    K, T = qf - qb, qf + qb
    sigma_e = K * np.log((T+K)/(T-K))
    R_e = sigma_e - 2 * K**2 / T
    B = np.zeros((N, NE))
    for k,(i,j) in enumerate(edges): B[i,k] = -1; B[j,k] = +1
    G = np.zeros((NE, N))
    for k,(i,j) in enumerate(edges): G[k,i] = fwd[k]; G[k,j] = -bwd[k]
    ones = np.ones((N,1))/np.sqrt(N)
    R = np.linalg.qr(np.eye(N) - ones @ ones.T)[0][:, :N-1]
    Gt = G @ R
    D_inv = np.diag(1.0/T)
    A_inner = Gt.T @ D_inv @ Gt
    M_T = D_inv - D_inv @ Gt @ np.linalg.pinv(A_inner) @ Gt.T @ D_inv
    u_,s_,vt_ = np.linalg.svd(B)
    rk = (s_ > max(B.shape)*s_.max()*np.finfo(float).eps).sum()
    C = vt_[rk:].T
    LT_eff = C.T @ M_T @ C
    Sigma_K = C @ np.linalg.inv(LT_eff) @ C.T
    KDK = float(K@D_inv@K); KMK = float(K@M_T@K)
    gamma_G = (KDK - KMK)/KDK if KDK > 0 else 0.0
    sigma = float(sigma_e.sum())
    return dict(K=K, T=T, sigma=sigma, sigma_e=sigma_e, R_e=R_e,
                Sigma_K=Sigma_K, M_T=M_T, D_inv=D_inv, NE=NE,
                gamma_G=gamma_G, KDK=KDK, KMK=KMK,
                fwd=np.asarray(fwd).copy(), bwd=np.asarray(bwd).copy(),
                p=p)

def estimators_over_subsets(net, n, subset_iter):
    K, sigma_e, R_e, Sigma_K = net['K'], net['sigma_e'], net['R_e'], net['Sigma_K']
    pe, cv, hb, mx = [], [], [], []
    for obs in subset_iter:
        obs = list(obs)
        K_o = K[obs]; S_o = Sigma_K[np.ix_(obs, obs)]
        per = float(sigma_e[obs].sum())
        cov = 2 * K_o @ np.linalg.pinv(S_o, rcond=1e-10) @ K_o
        hyb = cov + float(R_e[obs].sum())
        pe.append(per); cv.append(cov); hb.append(hyb); mx.append(max(per, hyb))
    return (np.array(pe), np.array(cv), np.array(hb), np.array(mx))

def sweep(net, sizes, max_subsets=400, full_enum_threshold=500):
    NE = net['NE']
    out = {k:([],[]) for k in ['pe','cv','hb','mx']}
    for n in sizes:
        from math import comb
        nc = comb(NE, n)
        if nc <= full_enum_threshold:
            it = list(itertools.combinations(range(NE), n))
        else:
            it = []
            seen = set()
            while len(it) < max_subsets:
                s = tuple(sorted(rng.choice(NE, n, replace=False)))
                if s not in seen:
                    seen.add(s); it.append(s)
        pe, cv, hb, mx = estimators_over_subsets(net, n, it)
        for key, arr in zip(['pe','cv','hb','mx'], [pe,cv,hb,mx]):
            out[key][0].append(np.mean(arr)); out[key][1].append(np.std(arr))
    return {k:(np.array(v[0]), np.array(v[1])) for k,v in out.items()}

# === Network A: 3x3 torus, vertex-transitive ===
def grid_3x3(c, u):
    edges = []
    def idx(r,c_): return 3*r + c_
    for r in range(3):
        for cc in range(3):
            edges.append((idx(r,cc), idx(r,(cc+1)%3)))
            edges.append((idx(r,cc), idx((r+1)%3,cc)))
    fwd = np.full(len(edges), c+u); bwd = np.full(len(edges), c-u)
    return 9, edges, fwd, bwd

N_A, edges_A, fwd_A, bwd_A = grid_3x3(1.0, 0.5)
net_A = network_objects(N_A, edges_A, fwd_A, bwd_A)
sizes_A = list(range(2, net_A['NE']+1, 2)) + [net_A['NE']]
sizes_A = sorted(set(sizes_A))
sweep_A = sweep(net_A, sizes_A, max_subsets=400)
print(f"A: torus, gamma_G={net_A['gamma_G']:.4f}, sigma={net_A['sigma']:.3f}")

# === Network B: 4-state, 5-edge, nonuniform; tune rates to hit target gamma_G ===
edges_B = [(0,1),(1,2),(2,0),(2,3),(3,0)]
def gamma_for_params(params):
    fwd = np.exp(params[:5]); bwd = np.exp(params[5:])
    try:
        net = network_objects(4, edges_B, fwd, bwd)
        if not np.all(net['T'] > np.abs(net['K'])): return None
        return net
    except Exception:
        return None

def find_rates_with_gamma(target, n_random=4000, refine=True):
    best = None
    for _ in range(n_random):
        params = rng.uniform(-1.5, 1.5, 10)
        net = gamma_for_params(params)
        if net is None: continue
        score = abs(net['gamma_G'] - target)
        if best is None or score < best[0]:
            best = (score, params, net)
    if refine:
        from scipy.optimize import minimize
        def obj(p):
            net = gamma_for_params(p)
            if net is None: return 1e3
            return (net['gamma_G'] - target)**2
        res = minimize(obj, best[1], method='Nelder-Mead',
                       options=dict(xatol=1e-5, fatol=1e-7, maxiter=2000))
        net = gamma_for_params(res.x)
        if net is not None:
            best = (abs(net['gamma_G']-target), res.x, net)
    return best[2]

target_gammas = [0.1, 0.2, 0.3, 0.4, 0.5]
nets_B = []
for tg in target_gammas:
    net = find_rates_with_gamma(tg)
    nets_B.append(net)
    print(f"B: target gamma={tg:.2f}  achieved={net['gamma_G']:.4f}  "
          f"sigma={net['sigma']:.3f}  ceiling/sigma={(2*net['KMK']+net['R_e'].sum())/net['sigma']:.3f}")

sizes_B = list(range(1, 6))
sweeps_B = [sweep(n, sizes_B) for n in nets_B]

# === Plot ===
fig = plt.figure(figsize=(7.0, 3.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.32)

# --- Panel A: torus ---
axA = fig.add_subplot(gs[0,0])
sa = np.array(sizes_A)
axA.errorbar(sa, sweep_A['mx'][0], yerr=sweep_A['mx'][1], fmt='*-',
             color='C4', lw=1.6, ms=7, capsize=2,
             label=r'$\mathcal{B}_{\mathcal{O}}^{\max}$')
axA.errorbar(sa, sweep_A['hb'][0], yerr=sweep_A['hb'][1], fmt='D-',
             color='C3', lw=1.4, ms=4, capsize=2,
             label=r'hybrid $\mathcal{B}_{\mathcal{O}}$')
axA.errorbar(sa, sweep_A['cv'][0], yerr=sweep_A['cv'][1], fmt='o-',
             color='C0', lw=1.4, ms=4, capsize=2,
             label='covariance only')
axA.errorbar(sa, sweep_A['pe'][0], yerr=sweep_A['pe'][1], fmt='s--',
             color='C2', lw=1.4, ms=4, capsize=2,
             label='per-edge nonlinear')
axA.axhline(net_A['sigma'], color='k', ls=':', lw=1.0,
            label=fr'$\sigma={net_A["sigma"]:.2f}$')
axA.set_xlabel(r"observed edges $|\mathcal{O}|$")
axA.set_ylabel("dissipation bound")
axA.set_xlim(0, net_A['NE']+1); axA.set_ylim(0, 1.1*net_A['sigma'])
axA.set_title(r"(a) $3\times 3$ torus, $\gamma_G\!=\!0$",
              fontsize=9)
axA.legend(fontsize=7.0, frameon=False, loc='lower right')

# Inset: 3x3 torus topology
axA_in = inset_axes(axA, width="32%", height="32%", loc='upper left',
                    borderpad=0.6)
pos_A = np.array([[c, r] for r in range(3) for c in range(3)], dtype=float)
def _idx(r,c): return 3*r + c
edges_A_draw = []
for r in range(3):
    for c in range(3):
        edges_A_draw.append((_idx(r,c), _idx(r,(c+1)%3)))
        edges_A_draw.append((_idx(r,c), _idx((r+1)%3,c)))
draw_network(axA_in, pos_A, edges_A_draw, node_size=22, lw=0.8)

# --- Panel B: nonuniform sweep over gamma ---
axB = fig.add_subplot(gs[0,1])
sb = np.array(sizes_B)
cmap = plt.cm.viridis
colors_g = [cmap(0.15 + 0.7*i/(len(target_gammas)-1)) for i in range(len(target_gammas))]
for i, (net, sw, gG) in enumerate(zip(nets_B, sweeps_B, target_gammas)):
    sigma = net['sigma']
    mx_norm = sw['mx'][0] / sigma
    mx_err  = sw['mx'][1] / sigma
    pe_norm = sw['pe'][0] / sigma
    pe_err  = sw['pe'][1] / sigma
    axB.errorbar(sb, mx_norm, yerr=mx_err, fmt='*-', color=colors_g[i],
                 lw=1.5, ms=8, capsize=2,
                 label=rf'$\gamma_G\!\approx\!{net["gamma_G"]:.2f}$:  $\mathcal{{B}}^{{\max}}$')
    axB.errorbar(sb, pe_norm, yerr=pe_err, fmt='s--', color=colors_g[i],
                 lw=1.0, ms=3, capsize=2, alpha=0.55)
    ceiling = (2*net['KMK'] + net['R_e'].sum())/sigma
    axB.axhline(ceiling, color=colors_g[i], ls=':', lw=0.9, alpha=0.6)
axB.axhline(1.0, color='k', ls=':', lw=1.0)
axB.set_xlabel(r"observed edges $|\mathcal{O}|$")
axB.set_ylabel(r"bound $/\,\sigma$")
axB.set_xlim(0.5, 5.5); axB.set_ylim(0, 1.06)
axB.set_title(r"(b) 4-state two-cycle, $\gamma_G$ sweep",
              fontsize=9)
# explanatory legend lines
from matplotlib.lines import Line2D
extra = [Line2D([0],[0], color='gray', ls='-', marker='*', ms=8, lw=1.5,
                label=r'$\mathcal{B}_{\mathcal{O}}^{\max}$ (solid)'),
         Line2D([0],[0], color='gray', ls='--', marker='s', ms=3, lw=1.0,
                label=r'$\sum_{\mathcal{O}} \sigma_e$ (dashed)'),
         Line2D([0],[0], color='gray', ls=':', lw=0.9,
                label=r'$\mathcal{B}_{\mathcal{E}}/\sigma$ (hybrid level)')]
handles_all, labels_all = axB.get_legend_handles_labels()
leg1 = axB.legend(handles=handles_all, labels=labels_all,
                  fontsize=6.5, frameon=False, loc='lower right',
                  ncol=1)
axB.add_artist(leg1)
axB.legend(handles=extra, fontsize=6.5, frameon=True, facecolor='white',
           edgecolor='none', framealpha=1, loc='upper left')

# Inset: 4-state two-cycle topology
axB_in = inset_axes(axB, width="26%", height="32%", loc='lower right',
                    bbox_to_anchor=(0.0, 0.18, 1.0, 1.0),
                    bbox_transform=axB.transAxes, borderpad=0.6)
pos_B = np.array([[-1.0, 0.0], [0.0, 0.7], [0.0, -0.7], [1.0, 0.0]])
draw_network(axB_in, pos_B, edges_B, node_size=28, lw=0.9)

plt.tight_layout()
plt.savefig("fig_main.pdf", dpi=200, bbox_inches='tight')
plt.savefig("fig_main.png", dpi=150, bbox_inches='tight')
print("saved fig_main.{pdf,png}")

# Persist plot data as CSV for reproducibility
import csv
with open("fig_main_panelA.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sigma_exact", net_A['sigma'], "gamma_G", net_A['gamma_G']])
    w.writerow(["|O|","per_edge_mean","per_edge_std","cov_mean","cov_std",
                "hybrid_mean","hybrid_std","Bmax_mean","Bmax_std"])
    for i, n in enumerate(sizes_A):
        w.writerow([n, sweep_A['pe'][0][i], sweep_A['pe'][1][i],
                    sweep_A['cv'][0][i], sweep_A['cv'][1][i],
                    sweep_A['hb'][0][i], sweep_A['hb'][1][i],
                    sweep_A['mx'][0][i], sweep_A['mx'][1][i]])
with open("fig_main_panelB.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["target_gamma","achieved_gamma","sigma","KDK","KMK",
                "ceiling_over_sigma","|O|","per_edge_norm","per_edge_std_norm",
                "cov_norm","cov_std_norm","hybrid_norm","hybrid_std_norm",
                "Bmax_norm","Bmax_std_norm"])
    for tg, net, sw in zip(target_gammas, nets_B, sweeps_B):
        s = net['sigma']; ceil = (2*net['KMK']+net['R_e'].sum())/s
        for i, n in enumerate(sizes_B):
            w.writerow([tg, net['gamma_G'], s, net['KDK'], net['KMK'], ceil, n,
                        sw['pe'][0][i]/s, sw['pe'][1][i]/s,
                        sw['cv'][0][i]/s, sw['cv'][1][i]/s,
                        sw['hb'][0][i]/s, sw['hb'][1][i]/s,
                        sw['mx'][0][i]/s, sw['mx'][1][i]/s])
# Network parameters for panel A torus
with open("fig_main_panelA_rates.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["edge_idx","i","j","fwd_rate","bwd_rate","K","T"])
    for k,(i,j) in enumerate(edges_A):
        w.writerow([k, i, j, fwd_A[k], bwd_A[k], net_A['K'][k], net_A['T'][k]])

# Network parameters for panel B (recover w_ij from q_ij = p_i w_ij)
with open("fig_main_panelB_rates.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["target_gamma","achieved_gamma","edge_idx","i","j",
                "qf","qb","K","T","w_ij","w_ji","p_i","p_j"])
    for tg, net in zip(target_gammas, nets_B):
        # rebuild stationary distribution from K,T,edges to recover w
        # use the original W: just resave rates from network_objects build
        # we need p — re-derive from net. We didn't store p; recompute using fluxes.
        # q_ij = (T+K)/2, q_ji = (T-K)/2; w_ij = q_ij/p_i, w_ji = q_ji/p_j
        # solve B p = 0 normalization from edges; here just rebuild
        K, T, p_ = net['K'], net['T'], net['p']
        qf = (T+K)/2; qb = (T-K)/2
        for k,(i,j) in enumerate(edges_B):
            w.writerow([tg, net['gamma_G'], k, i, j, qf[k], qb[k], K[k], T[k],
                        net['fwd'][k], net['bwd'][k], p_[i], p_[j]])
print("saved fig_main_panel{A,B}.csv, fig_main_panel{A,B}_rates.csv")
