"""Step 1 of the kinesin example: generator + thermodynamic bookkeeping.

Deliverables:
  1. A general continuous-time Markov network engine (stationary distribution,
     one-way fluxes, edge current/traffic, Schnakenberg entropy production,
     cycle basis + cycle affinities, density-contracted metric M_T, covariance
     Sigma_K).
  2. Validation of the engine on an analytically solvable biased 3-state ring.
  3. The Liepelt-Lipowsky low-product-concentration kinesin topology
     (6 states; hexagon 1-2-3-4-5-6 plus the mechanical chord 2-5).
     Mechanical step = chord; chemical = hexagon.
  4. The robust, rate-INDEPENDENT structural result: which observed edge sets
     are cycle-observing, hence when displacement-only observation can certify
     the full quadratic cycle cost and when the futile (hexagon) cycle stays hidden.

The rates below use the Carter--Cross parameter row [15] in Table I of
Liepelt and Lipowsky, PRL 98, 258102 (2007), with the same balance
relations and load factors. Both papers discuss the 7-state extension;
the 6-state model used here is the low-[ADP], low-[P_i] sector.
We retain rounded kappa_65 and enforce exact balance through kappa_16,
rather than retaining rounded kappa_16 and deriving kappa_65 as in Table I.
"""
import numpy as np

# ============================================================ engine
class MarkovNet:
    def __init__(self, n_states, edges, w_fwd, w_bwd):
        self.N = n_states
        self.edges = list(edges)              # undirected, oriented i->j
        self.NE = len(self.edges)
        self.wf = np.asarray(w_fwd, float)
        self.wb = np.asarray(w_bwd, float)
        self._build()

    def _build(self):
        N, E = self.N, self.edges
        W = np.zeros((N, N))
        for k, (i, j) in enumerate(E):
            W[i, j] += self.wf[k]; W[j, i] += self.wb[k]
        for i in range(N):
            W[i, i] = -W[i].sum()
        self.W = W
        # stationary distribution
        A = np.vstack([W.T, np.ones(N)]); b = np.zeros(N + 1); b[-1] = 1.0
        self.p = np.linalg.lstsq(A, b, rcond=None)[0]
        # one-way fluxes, current, traffic
        self.qf = np.array([self.p[i]*self.wf[k] for k,(i,j) in enumerate(E)])
        self.qb = np.array([self.p[j]*self.wb[k] for k,(i,j) in enumerate(E)])
        self.K = self.qf - self.qb
        self.T = self.qf + self.qb
        # edge force and Schnakenberg entropy production
        self.F = np.log(self.qf / self.qb)
        self.sigma_e = self.K * self.F
        self.sigma = float(self.sigma_e.sum())
        # incidence and cycle basis
        B = np.zeros((N, self.NE))
        for k, (i, j) in enumerate(E):
            B[i, k] = -1; B[j, k] = +1
        self.B = B
        _, s, vt = np.linalg.svd(B)
        self.rank_B = int((s > 1e-9).sum())
        self.C = vt[self.rank_B:].T            # NE x r
        self.r = self.C.shape[1]
        # density-contracted traffic metric M_T and covariance Sigma_K
        G = np.zeros((self.NE, N))
        for k, (i, j) in enumerate(E):
            G[k, i] = self.wf[k]; G[k, j] = -self.wb[k]
        ones = np.ones((N, 1)) / np.sqrt(N)
        Rb = np.linalg.qr(np.eye(N) - ones @ ones.T)[0][:, :N-1]
        Gt = G @ Rb
        Dinv = np.diag(1.0 / self.T)
        self.M = Dinv - Dinv @ Gt @ np.linalg.pinv(Gt.T @ Dinv @ Gt) @ Gt.T @ Dinv
        self.SigK = self.C @ np.linalg.inv(self.C.T @ self.M @ self.C) @ self.C.T
        self.gammaG = float((self.K @ Dinv @ self.K - self.K @ self.M @ self.K)
                            / (self.K @ Dinv @ self.K))

    def cycle_affinities(self):
        """Affinity (sum of edge force) around each fundamental cycle."""
        return self.C.T @ self.F

    def cycle_observing(self, obs):
        """True iff P_obs restricted to cycle space is injective (rank r)."""
        PO = np.zeros((len(obs), self.NE)); PO[range(len(obs)), obs] = 1.0
        return int(np.linalg.matrix_rank(PO @ self.C)) == self.r

    def covariance_bound(self, obs):
        """2 K_O^T Sigma_O^+ K_O  (the certified quadratic energy on O)."""
        SigO = self.SigK[np.ix_(obs, obs)]
        return 2 * self.K[obs] @ np.linalg.pinv(SigO) @ self.K[obs]


# ============================================================ 1) engine validation
def validate_ring():
    # biased 3-ring: clockwise rate a, counterclockwise b; uniform p=1/3.
    a, b = 2.0, 0.5
    edges = [(0, 1), (1, 2), (2, 0)]
    net = MarkovNet(3, edges, [a, a, a], [b, b, b])
    # analytic: p=1/3; K=(a-b)/3; T=(a+b)/3 per edge; sigma=3*K*ln(a/b)
    K_an = (a - b) / 3
    sig_an = 3 * K_an * np.log(a / b)
    print("[engine validation: biased 3-ring]")
    print(f"  p (num)      = {np.round(net.p,4)}  (analytic 1/3 = {1/3:.4f})")
    print(f"  K per edge   = {np.round(net.K,4)}  (analytic {K_an:.4f})")
    print(f"  sigma (num)  = {net.sigma:.6f}  (analytic {sig_an:.6f})")
    print(f"  match: {np.isclose(net.sigma, sig_an) and np.allclose(net.p, 1/3)}\n")


# ============================================================ 2) Liepelt-Lipowsky topology
def ll_edge_rates(ATP=1000.0, ADP=1.0, Pi=1.0, F=0.0):
    """Balance-consistent Liepelt-Lipowsky 6-state kinesin parametrization.

    Liepelt & Lipowsky, PRL 98, 258102 (2007), Table I, dataset [15]
    (Carter-Cross), load distribution factor theta = 0.65.  Concentrations in
    micromolar; rates in 1/s; binding constants in 1/(uM s); force F in pN.

    omega_ij = kappa_ij * I_ij([X]) * Phi_ij(F),  Phi_ij(0) = 1.
      I_ij = [X] if transition i->j binds species X, else 1.
      mechanical chord 2-5:  Phi_25 = exp(-theta*fT), Phi_52 = exp((1-theta)*fT)
      chemical:              Phi_ij = 2/(1+exp(chi_ij*fT)),  chi_ij = chi_ji
      fT = l*F/kT,  l = 8 nm,  kT = 4.1 pN nm.
    State map 1..6 -> 0..5.  Head symmetry (1<->4, 2<->5, 3<->6) sets the
    B-cycle constants from the F-cycle. We keep rounded kappa_65=0.02 and
    derive kappa_16=0.0255102041 by balance (3), with K_eq=4.9e11 uM;
    Table I instead marks kappa_65 as derived and supplies kappa_16=0.02.
    """
    theta = 0.65
    fT = (8.0 / 4.1) * F                        # l F / kT, dimensionless
    Keq = 4.9e11                                # concentration convention of Table I

    # Table I, row [15], with the balance/rounding convention above.
    k12, k21 = 2.0, 100.0                       # ATP binding / unbinding
    k25, k52 = 3.0e5, 0.24                      # forward / backward step
    k56, k65 = 100.0, 0.02                      # (5->6) / ADP binding (6->5)
    k61 = 100.0                                 # ATP hydrolysis (6->1)
    k16 = (k25*k12*k56*k61) / (k52*k21*k65*Keq) # synthesis, F-cycle balance
    k54 = (k52/k25)**2 * k21                     # B-cycle balance (paper Eq. 3)

    # chi per undirected edge (chi_ij = chi_ji); 0 where unspecified
    chi_e = {(0,1):0.25,(3,4):0.25,(1,2):0.15,(2,3):0.15,(4,5):0.15,(5,0):0.15}

    # directed specification: edge (i,j) -> (kappa_fwd, Xfwd, kappa_bwd, Xbwd)
    # X in {'ATP','ADP','P',None}; orientation i->j as listed.
    ATPc, ADPc, Pc = ATP, ADP, Pi
    spec = {
        (0,1): (k12, 'ATP', k21, None),         # 1->2 ATP bind / 2->1 unbind
        (1,2): (k56, None,  k65, 'ADP'),        # 2->3 (~5->6) / 3->2 ADP bind
        (2,3): (k61, None,  k16, 'P'),          # 3->4 hydrolysis / 4->3 synth
        (3,4): (k12, 'ATP', k54, None),         # 4->5 ATP bind / 5->4 (balance)
        (4,5): (k56, None,  k65, 'ADP'),        # 5->6 / 6->5 ADP bind
        (5,0): (k61, None,  k16, 'P'),          # 6->1 hydrolysis / 1->6 synth
        (1,4): (k25, None,  k52, None),         # chord 2->5 / 5->2 mechanical
    }
    conc = {'ATP':ATPc,'ADP':ADPc,'P':Pc,None:1.0}

    edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(1,4)]
    mech_edge = 6
    chem_edges = [0,1,2,3,4,5]
    wf = np.zeros(len(edges)); wb = np.zeros(len(edges))
    for e,(i,j) in enumerate(edges):
        kf, Xf, kb, Xb = spec[(i,j)]
        if e == mech_edge:
            Phi_f, Phi_b = np.exp(-theta*fT), np.exp((1-theta)*fT)
        else:
            c = chi_e.get((i,j), 0.0)
            Phi_f = Phi_b = 2.0/(1.0+np.exp(c*fT))
        wf[e] = kf * conc[Xf] * Phi_f
        wb[e] = kb * conc[Xb] * Phi_b
    return 6, edges, wf, wb, mech_edge, chem_edges


def ll_mechanical_current(ATP=1000.0, ADP=1.0, Pi=1.0, F=0.0):
    """Stationary current on the 2<->5 mechanical stepping transition."""
    N, edges, wf, wb, mech, chem = ll_edge_rates(ATP, ADP, Pi, F)
    return MarkovNet(N, edges, wf, wb).K[mech]


def ll_stall_force(ATP=1000.0, ADP=1.0, Pi=1.0, lo=0.0, hi=10.0, tol=1e-11):
    """Force where the net mechanical current vanishes for the 6-state model."""
    klo = ll_mechanical_current(ATP, ADP, Pi, lo)
    khi = ll_mechanical_current(ATP, ADP, Pi, hi)
    while klo * khi > 0 and hi < 40.0:
        hi *= 1.5
        khi = ll_mechanical_current(ATP, ADP, Pi, hi)
    if klo * khi > 0:
        raise RuntimeError("stall force is not bracketed")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        kmid = ll_mechanical_current(ATP, ADP, Pi, mid)
        if abs(kmid) < tol:
            return mid
        if klo * kmid <= 0:
            hi, khi = mid, kmid
        else:
            lo, klo = mid, kmid
    return 0.5 * (lo + hi)


def analyze_ll():
    N, edges, wf, wb, mech, chem = ll_edge_rates()
    net = MarkovNet(N, edges, wf, wb)
    print("[Liepelt-Lipowsky kinesin network]  (6-state Table-I Carter-Cross rates)")
    print(f"  states={net.N}, edges={net.NE}, cycle rank r={net.r}")
    print(f"  stationary p = {np.round(net.p,3)}")
    print(f"  total sigma  = {net.sigma:.4f} kT/s   gamma_G = {net.gammaG:.4f}")
    # physical cycle affinities F, B, D vs the L-L balance predictions
    fe = net.F                                  # edge forces ln(q+/q-)
    # edge index map: 0:(0,1) 1:(1,2) 2:(2,3) 3:(3,4) 4:(4,5) 5:(5,0) 6:chord
    A_F = fe[0] + fe[6] + fe[4] + fe[5]         # 1->2->5->6->1
    A_B = fe[3] - fe[6] + fe[1] + fe[2]         # 4->5->2->3->4
    A_D = A_F + A_B                             # hexagon
    ATP, ADP, Pi = 1000.0, 1.0, 1.0
    dmu = np.log(4.9e11 * ATP / (ADP*Pi))
    print(f"  Delta_mu = {dmu:.3f} kT   (ATP={ATP}, ADP={ADP}, Pi={Pi} uM)")
    print(f"  cycle affinities  A(F)={A_F:.3f}  A(B)={A_B:.3f}  A(D)={A_D:.3f}")
    print(f"  L-L predictions   A(F)=A(B)=Delta_mu={dmu:.3f}, A(D)=2*Delta_mu={2*dmu:.3f}")
    print(f"  thermodynamics OK: {np.allclose([A_F,A_B,A_D],[dmu,dmu,2*dmu],atol=1e-2)}")
    print(f"  mechanical edge (chord 2-5): index {mech}, "
          f"K={net.K[mech]:+.4f}, force f={net.F[mech]:+.3f}")
    print()

    # ---- the robust, rate-INDEPENDENT structural result ----
    print("[cycle-observability: when can a displacement observer certify all?]")
    obs_mech = [mech]
    print(f"  observe mechanical step only {obs_mech}: "
          f"cycle-observing = {net.cycle_observing(obs_mech)}")
    # add one chemical (hexagon) edge -> should span cycle space
    obs_mech_chem = [mech, chem[1]]
    print(f"  observe step + one chemical edge {obs_mech_chem}: "
          f"cycle-observing = {net.cycle_observing(obs_mech_chem)}")
    obs_all = list(range(net.NE))
    print(f"  observe all edges: cycle-observing = {net.cycle_observing(obs_all)}")
    print()

    # ---- what each observation certifies ----
    full = 2 * float(net.K @ net.M @ net.K)
    print("[certified quadratic energy 2 K_O^T Sigma_O^+ K_O]")
    print(f"  full-network 2 K^T M_T K          = {full:.4f}")
    for label, obs in [("mechanical step only", obs_mech),
                       ("step + one chemical",  obs_mech_chem),
                       ("all edges",            obs_all)]:
        val = net.covariance_bound(obs)
        certifies_full = np.isclose(val, full)
        print(f"  {label:22s}: {val:.4f}   "
              f"(certifies full energy: {certifies_full})")
    print("\n  -> mechanical-only observation is NOT cycle-observing: the futile")
    print("     hexagon cycle carries no mechanical edge, so its dissipation is")
    print("     invisible to a displacement detector.  Adding one chemical edge")
    print("     restores cycle observability and certifies the full energy.")


def force_sweep():
    """Locate the regime where displacement-only observation fails.

    Near the stall force the forward and backward stepping cycles balance, so
    the NET current on the mechanical edge vanishes while ATP is still
    hydrolyzed (futile cycling).  A displacement detector then certifies almost
    none of the dissipation, but observing the step plus one chemical
    transition certifies the full cycle energy.
    """
    Fstall = ll_stall_force(1000., 1., 1.)
    print(f"\n[load-force sweep: 1 mM ATP, 1 uM ADP/Pi]  "
          f"(kinesin stall {Fstall:.2f} pN)")
    print(" F[pN]   sigma     full M_T   step-only   step+chem   step/full")
    for F in [0, 2, 4, 6, Fstall, 8, 10]:
        N, edges, wf, wb, mech, chem = ll_edge_rates(1000., 1., 1., F)
        net = MarkovNet(N, edges, wf, wb)
        full = 2*float(net.K @ net.M @ net.K)
        mo = net.covariance_bound([mech])
        sc = net.covariance_bound([mech, chem[1]])
        print(f"  {F:4.1f}  {net.sigma:8.1f}  {full:8.2f}   {mo:8.3f}    "
              f"{sc:8.2f}    {mo/full:6.3f}")
    print("  -> at stall the displacement-only certificate collapses to ~0;")
    print("     step + one chemical transition recovers the full cycle energy.")


if __name__ == "__main__":
    validate_ring()
    analyze_ll()
    force_sweep()
