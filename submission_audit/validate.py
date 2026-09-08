"""Independent numerical checks for PRL_draftv5; does not regenerate figures.

Run from the code directory: python submission_audit/validate.py
Uses a Poisson-equation martingale covariance, independently of M_T.
"""
import ast
import csv
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kinesin", ROOT / "kinesin_step1.py")
ks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ks)


def functions_only(filename, names, **namespace):
    # Import selected definitions without running top-level simulations/plots.
    tree = ast.parse((ROOT / filename).read_text())
    tree.body = [n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace["np"] = np
    exec(compile(tree, str(ROOT / filename), "exec"), namespace)
    return namespace


def poisson_objects(n, edges, wf, wb):
    """Solve W h = K - j; sum stationary quadratic variations of jumps."""
    W = np.zeros((n, n))
    j = np.zeros((n, len(edges)))
    for e, (i, k) in enumerate(edges):
        W[i, k] += wf[e]
        W[k, i] += wb[e]
        j[i, e] += wf[e]
        j[k, e] -= wb[e]
    W -= np.diag(W.sum(axis=1))
    A = W.T.copy()
    A[-1] = 1
    rhs = np.zeros(n)
    rhs[-1] = 1
    p = np.linalg.solve(A, rhs)
    K = p @ j
    P = np.outer(np.ones(n), p)
    h = np.linalg.solve(W + P, K[None, :] - j)
    S = np.zeros((len(edges), len(edges)))
    T = np.zeros(len(edges))
    left = np.zeros((len(edges), n))
    for e, (i, k) in enumerate(edges):
        d = np.eye(len(edges))[e] + h[k] - h[i]
        T[e] = p[i] * wf[e] + p[k] * wb[e]
        S += T[e] * np.outer(d, d)
        left[e, k] += p[i] * wf[e]
        left[e, i] -= p[k] * wb[e]
    return dict(W=W, p=p, K=K, T=T, S=S, j=j, left=left, P=P)


def finite_covariance(o, tau):
    """Exact tau*Cov(N(tau)/tau) for stationary jump counts."""
    W, P = o["W"], o["P"]
    Z = np.linalg.inv(W + P) - P
    H = Z @ Z @ (expm(tau * W) - np.eye(len(W))) - tau * Z
    term = o["left"] @ H @ o["j"] / tau
    return np.diag(o["T"]) + term + term.T


def relerr(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def quadratic(K, S, obs):
    return float(2 * K[obs] @ np.linalg.pinv(S[np.ix_(obs, obs)]) @ K[obs])


out = {}
scripts = list(ROOT.glob("*.py"))
for script in scripts:
    ast.parse(script.read_text(), filename=str(script))
out["python_syntax_checks"] = len(scripts)
ring = ks.MarkovNet(3, [(0, 1), (1, 2), (2, 0)], [2.] * 3, [.5] * 3)
np.testing.assert_allclose(ring.SigK, np.full((3, 3), 2.5 / 9), atol=1e-12)
out["ring_covariance_check"] = "passed"

edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 0)]
wf = np.array([2., 1.3, 1.7, 1.1, .9])
wb = np.array([1., .7, 1., .6, 1.4])
finite = functions_only("make_finite_sample_figure.py", {"stationary", "exact", "sim"},
                        N=4, NE=5, edges=edges, fwd=wf, bwd=wb)
syn = finite["exact"]()
ind = poisson_objects(4, edges, wf, wb)
np.testing.assert_allclose(syn["SigK"], ind["S"], rtol=1e-9, atol=1e-11)
O = [2, 4]
cov = quadratic(ind["K"], ind["S"], O)
Rsum = float(syn["R"][O].sum())
sigma = syn["sigma"]
out["synthetic"] = dict(sigma=sigma, covariance=cov, hybrid=cov + Rsum,
    covariance_over_sigma=cov / sigma, hybrid_over_sigma=(cov + Rsum) / sigma,
    per_edge_over_sigma=float(syn["sig_e"][O].sum() / sigma),
    best_single_edge_over_sigma=max(quadratic(ind["K"], ind["S"], [e])
                                   for e in O) / sigma,
    fixed_block_limit={str(tau): quadratic(ind["K"], finite_covariance(ind, tau), O)
                       + Rsum for tau in [8, 16, 32, 40, 64, 128, 1000]})

# Verify the finite-window formula with a separate tilted-generator derivative.
tau_check, tilt_step = 40., 1e-4
def finite_scalar_variance(direction):
    values = []
    for z in [-tilt_step, 0., tilt_step]:
        tilted = ind["W"].copy()
        for edge, (i, j) in enumerate(edges):
            tilted[i, j] *= np.exp(z * direction[edge])
            tilted[j, i] *= np.exp(-z * direction[edge])
        values.append(np.log(ind["p"] @ expm(tau_check * tilted) @ np.ones(4)))
    return (values[0] - 2 * values[1] + values[2]) / (tilt_step**2 * tau_check)

for direction in [np.eye(5)[2], np.eye(5)[4], np.eye(5)[2] + np.eye(5)[4]]:
    np.testing.assert_allclose(finite_scalar_variance(direction),
        direction @ finite_covariance(ind, tau_check) @ direction, rtol=1e-5)
out["finite_window_tilted_generator_check"] = "passed"

main = functions_only("make_main_figure.py",
    {"stationary", "network_objects", "estimators_over_subsets", "grid_3x3"})
torus_args = main["grid_3x3"](1., .5)
torus = main["network_objects"](*torus_args)
ind_torus = poisson_objects(*torus_args)
np.testing.assert_allclose(torus["Sigma_K"], ind_torus["S"], atol=1e-11)
with (ROOT / "fig_main_panelA.csv").open() as f:
    reader = csv.reader(f)
    next(reader)
    rows = list(csv.DictReader(f))
row12 = next(row for row in rows if int(row["|O|"]) == 12)
out["torus_saved_12_of_18_fraction"] = float(row12["Bmax_mean"]) / torus["sigma"]
all12 = main["estimators_over_subsets"](
    torus, 12, itertools.combinations(range(18), 12))[-1]
out["torus_all_12_of_18_fraction"] = float(all12.mean() / torus["sigma"])
tm = ks.MarkovNet(*torus_args)
S_raw = tm.C @ np.linalg.inv(tm.C.T @ np.diag(1 / tm.T) @ tm.C) @ tm.C.T
out["current_specific_unscreening_is_not_global"] = dict(
    gamma=tm.gammaG, observed=[0],
    M_completion=quadratic(tm.K, tm.SigK, [0]) / 2,
    raw_completion=quadratic(tm.K, S_raw, [0]) / 2)

with (ROOT / "fig_main_panelB_rates.csv").open() as f:
    rate_rows = list(csv.DictReader(f))
with (ROOT / "fig_main_panelB.csv").open() as f:
    data_rows = list(csv.DictReader(f))
checks = []
for gamma in sorted({r["target_gamma"] for r in rate_rows}):
    rows = [r for r in rate_rows if r["target_gamma"] == gamma]
    wf_b = np.array([float(r["w_ij"]) for r in rows])
    wb_b = np.array([float(r["w_ji"]) for r in rows])
    net = main["network_objects"](4, edges, wf_b, wb_b)
    independent = poisson_objects(4, edges, wf_b, wb_b)
    np.testing.assert_allclose(net["Sigma_K"], independent["S"], atol=1e-10)
    full = main["estimators_over_subsets"](net, 5, [range(5)])
    checks.append(dict(gamma=float(gamma), covariance_relative_error=relerr(
        net["Sigma_K"], independent["S"]), hybrid_full=float(full[2][0] / net["sigma"]),
        Bmax_full=float(full[3][0] / net["sigma"])))
    for n in range(1, 6):
        vals = main["estimators_over_subsets"](
            net, n, itertools.combinations(range(5), n))
        row = next(r for r in data_rows if r["target_gamma"] == gamma
                   and int(r["|O|"]) == n)
        for key, arr in zip(["per_edge_norm", "cov_norm", "hybrid_norm", "Bmax_norm"], vals):
            np.testing.assert_allclose(arr.mean() / net["sigma"], float(row[key]), atol=1e-10)
out["panel_B_reproduction"] = checks

rng = np.random.default_rng(82026)
max_error = 0.
for _ in range(100):
    f, b = np.exp(rng.uniform(-2, 2, (2, 5)))
    net = ks.MarkovNet(4, edges, f, b)
    independent = poisson_objects(4, edges, f, b)
    max_error = max(max_error, relerr(net.SigK, independent["S"]))
    np.testing.assert_allclose(net.SigK, independent["S"], rtol=1e-7, atol=1e-9)
    values = {}
    R = net.sigma_e - 2 * net.K**2 / net.T
    for size in range(1, 6):
        for subset in itertools.combinations(range(5), size):
            obs = list(subset)
            val = max(float(net.sigma_e[obs].sum()),
                      quadratic(net.K, net.SigK, obs) + R[obs].sum())
            assert val <= net.sigma + 1e-8
            if net.cycle_observing(obs):
                np.testing.assert_allclose(quadratic(net.K, net.SigK, obs),
                                           2 * net.K @ net.M @ net.K, atol=1e-8)
            for smaller, previous in values.items():
                if set(smaller).issubset(subset):
                    assert previous <= val + 1e-8
            values[subset] = val
out["random_network_checks"] = dict(networks=100, max_covariance_relative_error=max_error,
                                    bounds_monotonicity_and_observability="passed")

stall = ks.ll_stall_force()
errors = []
dmu = np.log(4.9e11 * 1000)
for force in np.linspace(0, 10, 41):
    n, ll_edges, f, b, mech, chem = ks.ll_edge_rates(F=force)
    net = ks.MarkovNet(n, ll_edges, f, b)
    independent = poisson_objects(n, ll_edges, f, b)
    errors.append(relerr(net.SigK, independent["S"]))
    np.testing.assert_allclose(net.SigK, independent["S"], rtol=1e-6, atol=1e-7)
    A_F = net.F[[0, 6, 4, 5]].sum()
    A_B = net.F[[3, 1, 2]].sum() - net.F[6]
    np.testing.assert_allclose([A_F, A_B, A_F + A_B],
                              [dmu - 8 * force / 4.1, dmu + 8 * force / 4.1, 2 * dmu],
                              atol=1e-8)
    hydrolysis = net.K[2] + net.K[5]
    np.testing.assert_allclose(net.sigma, dmu * hydrolysis - (8 * force / 4.1) * net.K[mech],
                              atol=1e-7)
n, ll_edges, f, b, mech, chem = ks.ll_edge_rates(F=stall)
net = ks.MarkovNet(n, ll_edges, f, b)
ind_ll = poisson_objects(n, ll_edges, f, b)
out["kinesin"] = dict(stall_pN=stall, sigma_stall=net.sigma,
    covariance_bound_stall=quadratic(net.K, net.SigK, [mech, chem[1]]),
    mechanical_bound_stall=quadratic(net.K, net.SigK, [mech]),
    hydrolysis_per_second=float(net.K[2] + net.K[5]),
    max_covariance_relative_error=max(errors),
    fixed_50s_block_limit=quadratic(ind_ll["K"], finite_covariance(ind_ll, 50), [mech, chem[1]]),
    covariance_eigenvalues=np.linalg.eigvalsh(net.SigK[np.ix_([mech, chem[1]], [mech, chem[1]])]).tolist())

# Table I marks k65 as balance-derived and k16 as the supplied 0.02 value.
table_k16 = .02
table_k65 = 3e5 * 2 * 100 * 100 / (.24 * 100 * table_k16 * 4.9e11)
code_k16 = 3e5 * 2 * 100 * 100 / (.24 * 100 * .02 * 4.9e11)
def table_network(force):
    n, e, f, b, m, c = ks.ll_edge_rates(F=force)
    b[[1, 4]] *= table_k65 / .02
    b[[2, 5]] *= table_k16 / code_k16
    return ks.MarkovNet(n, e, f, b)

table_stall = brentq(lambda force: table_network(force).K[6], 0, 10)
tn = table_network(table_stall)
out["kinesin_rate_provenance"] = dict(code_k65=.02, code_k16=code_k16,
    table_fixed_k16=table_k16, table_balance_derived_k65=table_k65,
    table_convention_stall_pN=table_stall, table_convention_sigma=tn.sigma,
    table_convention_covariance_bound=tn.covariance_bound([6, 1]))
with np.load(ROOT / "fig_kinesin_sim_stall_exact.npz") as cache:
    means, errs = cache["sc_mean"], cache["sc_err"]
    plot_tree = ast.parse((ROOT / "make_kinesin_figure.py").read_text())
    limit_call = next(node for node in ast.walk(plot_tree)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                      and isinstance(node.func.value, ast.Name)
                      and node.func.value.id == "axin" and node.func.attr == "set_ylim")
    scope = dict(sc_lower=means - errs, sc_upper=means + errs,
                 exact_sc=out["kinesin"]["covariance_bound_stall"], do_mean=cache["do_mean"])
    limits = [eval(compile(ast.Expression(arg), "<inset limit>", "eval"), scope)
              for arg in limit_call.args]
    assert limits[0] <= (means - errs).min() and limits[1] >= (means + errs).max()
    assert {"wf", "wb", "edges", "observed", "initial_p", "rcond", "seed_base",
            "simulator_sha256"}.issubset(cache.files)
    out["kinesin_inset_check"] = dict(status="passed", ylim=limits,
                                    errorbar_tops=(means + errs).tolist())

simulator = functions_only("make_kinesin_figure.py", {"gillespie_blocks"}, net=net, edges=ll_edges, NE=7)
blocks = simulator["gillespie_blocks"](f, b, 50., 1000, seed=882026)
S_emp = 50 * np.cov(blocks[:, [mech, chem[1]]].T)
S_theory = net.SigK[np.ix_([mech, chem[1]], [mech, chem[1]])]
out["kinesin_trajectory_check"] = dict(blocks=1000, block_seconds=50,
    empirical_covariance=S_emp.tolist(), theoretical_covariance=S_theory.tolist(),
    covariance_relative_error=relerr(S_emp, S_theory),
    empirical_observed_current=blocks.mean(axis=0)[[mech, chem[1]]].tolist(),
    theoretical_observed_current=net.K[[mech, chem[1]]].tolist())

print(json.dumps(out, indent=2))
