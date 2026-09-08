"""Finite-sample test of the hybrid bound + cycle-observability certification.

Step 1 (this file): exact theory on a 4-state two-cycle network, choose a
cycle-observing observed set O (a cotree, 2 of 5 edges), and validate that the
simulated long-time edge-current covariance matches the level-2.5 theory
Sigma_K = C (C^T M_T C)^{-1} C^T.  Prints the key numbers so we can confirm the
story before drawing the figure.
"""
import numpy as np

rng = np.random.default_rng(7)

# ---- network: 4 states, 5 edges, two cycles sharing edge (2,0) ----
edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 0)]
N, NE = 4, len(edges)
# nonuniform rates (forward, backward) -> generic gamma_G > 0
fwd = np.array([2.0, 1.3, 1.7, 1.1, 0.9])
bwd = np.array([1.0, 0.7, 1.0, 0.6, 1.4])

def stationary(W):
    A = np.vstack([W.T, np.ones(N)]); b = np.zeros(N + 1); b[-1] = 1.0
    return np.linalg.lstsq(A, b, rcond=None)[0]

def objects(fwd, bwd):
    W = np.zeros((N, N))
    for k, (i, j) in enumerate(edges):
        W[i, j] += fwd[k]; W[j, i] += bwd[k]
    for i in range(N):
        W[i, i] = -W[i].sum()
    p = stationary(W)
    qf = np.array([p[i] * fwd[k] for k, (i, j) in enumerate(edges)])
    qb = np.array([p[j] * bwd[k] for k, (i, j) in enumerate(edges)])
    K, T = qf - qb, qf + qb
    sig_e = K * np.log((T + K) / (T - K))
    R = sig_e - 2 * K**2 / T
    # incidence B (N x NE), divergence at j is +K_e for edge i->j
    B = np.zeros((N, NE))
    for k, (i, j) in enumerate(edges):
        B[i, k] = -1; B[j, k] = +1
    # cycle basis C = null(B)
    _, s, vt = np.linalg.svd(B)
    rank = (s > 1e-10).sum()
    C = vt[rank:].T                       # NE x r
    # density-response G (NE x N), restricted to sum-zero subspace
    G = np.zeros((NE, N))
    for k, (i, j) in enumerate(edges):
        G[k, i] = fwd[k]; G[k, j] = -bwd[k]
    ones = np.ones((N, 1)) / np.sqrt(N)
    Rb = np.linalg.qr(np.eye(N) - ones @ ones.T)[0][:, :N - 1]
    Gt = G @ Rb
    Dinv = np.diag(1.0 / T)
    M = Dinv - Dinv @ Gt @ np.linalg.pinv(Gt.T @ Dinv @ Gt) @ Gt.T @ Dinv
    SigK = C @ np.linalg.inv(C.T @ M @ C) @ C.T
    return dict(p=p, K=K, T=T, R=R, sig_e=sig_e, sigma=float(sig_e.sum()),
                C=C, M=M, Dinv=Dinv, SigK=SigK, W=W,
                gammaG=float((K @ Dinv @ K - K @ M @ K) / (K @ Dinv @ K)))

o = objects(fwd, bwd)
K, T, R, M, C, SigK = o['K'], o['T'], o['R'], o['M'], o['C'], o['SigK']
sigma = o['sigma']

# ---- choose observed set O = cotree (cycle-observing) ----
# spanning tree {(0,1),(1,2),(2,3)} = edges 0,1,3 ; cotree = edges 2,4
O = [2, 4]
H = [0, 1, 3]
PO = np.zeros((len(O), NE)); PO[range(len(O)), O] = 1.0
# cycle-observability check: P_O C must be square invertible (rank r)
POC = PO @ C
print(f"gamma_G = {o['gammaG']:.4f}")
print(f"cycle rank r = {C.shape[1]},  rank(P_O C) = {np.linalg.matrix_rank(POC)} "
      f"-> cycle-observing: {np.linalg.matrix_rank(POC) == C.shape[1]}")

# exact estimators
SigO = SigK[np.ix_(O, O)]
cov_term = 2 * K[O] @ np.linalg.pinv(SigO) @ K[O]
KMK = 2 * float(K @ M @ K)
per_edge_obs = float(o['sig_e'][O].sum())
hybrid = cov_term + float(R[O].sum())
Bmax = max(per_edge_obs, hybrid)
print(f"\nexact sigma                         = {sigma:.4f}")
print(f"2 K^T M_T K (full density-contracted)= {KMK:.4f}")
print(f"cov term 2 K_O^T Sig_O^+ K_O         = {cov_term:.4f}  "
      f"(certification: equals 2K^T M_T K? {np.isclose(cov_term, KMK)})")
print(f"observed-edge nonlinear sum          = {per_edge_obs:.4f}")
print(f"hybrid B_O                           = {hybrid:.4f}")
print(f"operational B_O^max                  = {Bmax:.4f}  ({100*Bmax/sigma:.1f}% of sigma)")
print(f"  -> covariance certifies more than observed-edge sum: "
      f"{cov_term > per_edge_obs}")

# ---- Gillespie: validate simulated covariance vs theory ----
def gillespie_blocks(fwd, bwd, tau, nblocks, seed):
    g = np.random.default_rng(seed)
    out = [[] for _ in range(N)]
    for k, (i, j) in enumerate(edges):
        out[i].append((j, k, +1)); out[j].append((i, k, -1))
    rates = [np.array([fwd[k] if d == 1 else bwd[k] for (_, k, d) in out[s]])
             for s in range(N)]
    tot = np.array([r.sum() for r in rates])
    s = int(g.choice(N, p=objects(fwd, bwd)['p']))
    Kblk = np.zeros((nblocks, NE)); Tcnt = np.zeros(NE)
    for b in range(nblocks):
        nf = np.zeros(NE); nb = np.zeros(NE); t = 0.0
        while True:
            dt = g.exponential(1 / tot[s])
            if t + dt > tau:
                break
            t += dt
            dest, k, d = out[s][g.choice(len(out[s]), p=rates[s]/tot[s])]
            if d == 1: nf[k] += 1
            else:      nb[k] += 1
            s = dest
        Kblk[b] = (nf - nb) / tau; Tcnt += nf + nb
    return Kblk, Tcnt / (nblocks * tau)

tau = 40.0
Kblk, Temp = gillespie_blocks(fwd, bwd, tau, 4000, seed=1)
Sig_sim = np.cov(Kblk.T) * tau
print("\n--- covariance validation (cycle subspace, edges O) ---")
print("theory  Sigma_O =\n", np.round(SigO, 4))
print("sim     Sigma_O =\n", np.round(Sig_sim[np.ix_(O, O)], 4))
print(f"relative Frobenius error on O-block = "
      f"{np.linalg.norm(Sig_sim[np.ix_(O,O)]-SigO)/np.linalg.norm(SigO):.3f}")
print(f"K (theory) = {np.round(K,4)}")
print(f"K (sim)    = {np.round(Kblk.mean(0),4)}")
