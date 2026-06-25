import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import pickle
import time


def is_executable_file(path):
    """Return True when *path* exists as a regular executable file.

    This mirrors the POSIX shell-style executable check: directories and missing
    paths are not executable files, while regular files must have execute
    permission for the current process.
    """
    return os.path.isfile(path) and os.access(path, os.X_OK)


def maybe_handle_executable_check(argv=None):
    """Handle the lightweight executable-file check before optional heavy imports."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--check_executable', default=None)
    args, _ = parser.parse_known_args(argv)
    if args.check_executable is None:
        return
    if is_executable_file(args.check_executable):
        print(f'Executable file: {args.check_executable}')
        raise SystemExit(0)
    print(f'Not an executable file: {args.check_executable}')
    raise SystemExit(1)


if __name__ == "__main__":
    maybe_handle_executable_check(sys.argv[1:])


import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize, minimize_scalar
from scipy.linalg import null_space
from scipy.sparse.linalg import LinearOperator, cg
from tqdm import tqdm

class MVSKOracle:
    """Exact sample-oracle MVSK objective and derivatives."""

    def __init__(self, R, c1=0.2, c2=1.0, c3=0.2, c4=0.05):
        self.R = np.asarray(R, dtype=float)
        self.T, self.n = self.R.shape
        self.mu = self.R.mean(axis=0)
        self.A = self.R - np.ones((self.T, 1)) * self.mu[None, :]
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.c3 = float(c3)
        self.c4 = float(c4)

    def value(self, x):
        x = np.asarray(x, dtype=float)
        z = self.A @ x
        m1 = self.mu @ x
        psi = self.c2 * z ** 2 - self.c3 * z ** 3 + self.c4 * z ** 4
        return -self.c1 * m1 + psi.mean()

    def grad(self, x):
        x = np.asarray(x, dtype=float)
        z = self.A @ x
        z2 = z * z
        z3 = z2 * z
        g = -self.c1 * self.mu
        g += (2.0 * self.c2 / self.T) * (self.A.T @ z)
        g -= (3.0 * self.c3 / self.T) * (self.A.T @ z2)
        g += (4.0 * self.c4 / self.T) * (self.A.T @ z3)
        return g

    def hess_vec(self, v):
        v = np.asarray(v, dtype=float)
        Av = self.A @ v
        z = self.A @ self.current_x
        z2 = z * z
        hv = (2.0 * self.c2 / self.T) * (self.A.T @ Av)
        hv -= (6.0 * self.c3 / self.T) * (self.A.T @ (z * Av))
        hv += (12.0 * self.c4 / self.T) * (self.A.T @ (z2 * Av))
        return hv

    def third_action(self, u, v):
        """Third derivative bilinear action T_x(u, v)."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        Au = self.A @ u
        Av = self.A @ v
        z = self.A @ self.current_x
        tv = -(6.0 * self.c3 / self.T) * (self.A.T @ (Au * Av))
        tv += (24.0 * self.c4 / self.T) * (self.A.T @ (z * Au * Av))
        return tv

    def set_current_x(self, x):
        self.current_x = np.asarray(x, dtype=float)


def simplex_tangent_basis(n):
    """Orthornormal basis for the simplex tangent space 1^T v = 0."""
    U = null_space(np.ones((1, n)))
    return U.astype(float)


def project_to_simplex(x, eps=1e-12):
    """Simplex projection via Euclidean projection (for demo purposes)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    n = x.shape[0]
    if n == 0:
        return x

    # Simple projection onto the probability simplex.
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > eps
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = np.maximum(x - theta, 0.0)
    return w


def line_search(oracle, x, d, max_iter=30):
    """Feasible line search using a bounded one-dimensional minimization."""
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)

    # Max feasible step before violating x_i >= 0.
    if np.any(d < 0):
        alpha_max = np.min(-x[d < 0] / d[d < 0])
    else:
        alpha_max = 1.0
    alpha_max = max(alpha_max, 1e-8)

    def phi(alpha):
        y = x + alpha * d
        return oracle.value(y)

    res = minimize_scalar(phi, bounds=(0.0, alpha_max), method='bounded', options={'xatol': 1e-8, 'maxiter': max_iter})
    alpha_star = res.x
    return alpha_star, x + alpha_star * d


def exact_quartic_line_search(oracle, x, d, tau=0.0):
    """Exact feasible line search for the quartic MVSK objective.

    Along any simplex tangent direction d, f(x + alpha d) is a quartic polynomial.
    The minimizer on the feasible interval is therefore attained at an endpoint or
    at a real root of the cubic derivative.
    """
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)

    if np.linalg.norm(d) < 1e-14:
        return 0.0, x.copy()

    lower = float(tau)
    if np.any(d < 0):
        alpha_max = np.min((x[d < 0] - lower) / (-d[d < 0]))
    else:
        alpha_max = 1.0
    alpha_max = float(max(alpha_max, 0.0))
    if alpha_max <= 1e-14:
        return 0.0, x.copy()

    z = oracle.A @ x
    w = oracle.A @ d
    m0 = float(oracle.mu @ x)
    md = float(oracle.mu @ d)

    # Polynomial coefficients phi(alpha) = p0 + p1*a + p2*a^2 + p3*a^3 + p4*a^4.
    p0 = -oracle.c1 * m0 + np.mean(oracle.c2 * z**2 - oracle.c3 * z**3 + oracle.c4 * z**4)
    p1 = -oracle.c1 * md + np.mean(2 * oracle.c2 * z * w - 3 * oracle.c3 * z**2 * w + 4 * oracle.c4 * z**3 * w)
    p2 = np.mean(oracle.c2 * w**2 - 3 * oracle.c3 * z * w**2 + 6 * oracle.c4 * z**2 * w**2)
    p3 = np.mean(-oracle.c3 * w**3 + 4 * oracle.c4 * z * w**3)
    p4 = np.mean(oracle.c4 * w**4)

    candidates = [0.0, alpha_max]
    deriv = np.array([4 * p4, 3 * p3, 2 * p2, p1], dtype=float)
    scale = max(1.0, np.max(np.abs(deriv)))
    deriv = np.trim_zeros(deriv / scale, trim='f')
    if deriv.size > 1:
        for root in np.roots(deriv):
            if abs(root.imag) <= 1e-9:
                a = float(root.real)
                if -1e-10 <= a <= alpha_max + 1e-10:
                    candidates.append(min(max(a, 0.0), alpha_max))

    def poly(a):
        return (((p4 * a + p3) * a + p2) * a + p1) * a + p0

    alpha_star = min(candidates, key=poly)
    y = x + alpha_star * d
    y[np.abs(y) < 1e-14] = 0.0
    return float(alpha_star), y


def affine_normal_direction(oracle, x, U, lam=1e-6):
    """A practical reduced-coordinate YAND-style direction."""
    oracle.set_current_x(x)
    g = oracle.grad(x)
    g_bar = U.T @ g
    g_norm = np.linalg.norm(g_bar)
    if g_norm < 1e-8:
        return None

    # Reduced Hessian action for the tangent space.
    def H_red(v):
        return U.T @ oracle.hess_vec(U @ v)

    # Regularized Newton step in reduced coordinates.
    H = np.zeros((U.shape[1], U.shape[1]), dtype=float)
    for i in range(U.shape[1]):
        ei = np.zeros(U.shape[1])
        ei[i] = 1.0
        H[:, i] = H_red(ei)

    # Add a small shift to keep the system well-conditioned.
    reg = lam + 1e-8 * np.linalg.norm(H, ord=2)
    try:
        d_y = -np.linalg.solve(H + reg * np.eye(H.shape[0]), g_bar)
    except np.linalg.LinAlgError:
        d_y = -np.linalg.pinv(H + reg * np.eye(H.shape[0])) @ g_bar

    # Fallback to steepest descent if the direction is not descent.
    if g_bar @ d_y >= 0:
        d_y = -g_bar

    d = U @ d_y
    return d


def householder_frame(nu):
    """Return an orthonormal frame for the subspace orthogonal to nu."""
    nu = np.asarray(nu, dtype=float)
    nu_norm = np.linalg.norm(nu)
    if nu_norm <= 0:
        raise ValueError('nu must be nonzero')
    return null_space((nu / nu_norm)[None, :]).astype(float)


def affine_normal_householder_direction(oracle, x, U, lam=1e-6, include_logdet=True):
    """Direct reduced Householder/log-det affine-normal direction.

    This is the dense small/medium-scale implementation of the paper's reduced
    direction. The log-det correction is assembled from exact third-order
    directional actions through trace(H_T^{-1} dH_T).
    """
    oracle.set_current_x(x)
    g = oracle.grad(x)
    g_bar = U.T @ g
    g_norm = np.linalg.norm(g_bar)
    if g_norm < 1e-10:
        return None

    m = U.shape[1]
    if m <= 1:
        return -U @ (g_bar / g_norm)

    nu = g_bar / g_norm
    Q = householder_frame(nu)

    def H_red(v):
        return U.T @ oracle.hess_vec(U @ v)

    H = np.zeros((m, m), dtype=float)
    for i in range(m):
        ei = np.zeros(m)
        ei[i] = 1.0
        H[:, i] = H_red(ei)
    H = 0.5 * (H + H.T)

    h = Q.T @ H @ nu
    HT = Q.T @ H @ Q
    reg = lam + 1e-10 * max(1.0, np.linalg.norm(HT, ord=2))
    HT_reg = HT + reg * np.eye(HT.shape[0])

    a = np.zeros(Q.shape[1], dtype=float)
    if include_logdet and Q.shape[1] > 0:
        try:
            HT_inv = np.linalg.inv(HT_reg)
        except np.linalg.LinAlgError:
            HT_inv = np.linalg.pinv(HT_reg)
        for j in range(Q.shape[1]):
            qj = Q[:, j]
            dHT = np.zeros_like(HT)
            uqj = U @ qj
            for i in range(Q.shape[1]):
                qi = Q[:, i]
                col = U.T @ oracle.third_action(uqj, U @ qi)
                dHT[:, i] = Q.T @ col
            dHT = 0.5 * (dHT + dHT.T)
            a[j] = np.trace(HT_inv @ dHT)

    try:
        y = np.linalg.solve(HT_reg, h - (g_norm / max(1, m + 1)) * a)
    except np.linalg.LinAlgError:
        y = np.linalg.pinv(HT_reg) @ (h - (g_norm / max(1, m + 1)) * a)

    d_y = Q @ y - nu
    if g_bar @ d_y >= 0 or not np.all(np.isfinite(d_y)):
        d_y = -g_bar

    return U @ d_y


def tangent_project(v):
    """Project a vector onto the simplex tangent space {v: 1^T v = 0}."""
    v = np.asarray(v, dtype=float)
    return v - np.mean(v)


def matrix_free_tangent_direction(oracle, x, lam=1e-6, max_cg_iter=200, tol=1e-8):
    """Large-scale tangent Newton-like direction using Hessian-vector products only."""
    oracle.set_current_x(x)
    g_tan = tangent_project(oracle.grad(x))
    if np.linalg.norm(g_tan) < tol:
        return None

    n = x.shape[0]
    reg = lam

    def matvec(v):
        v = tangent_project(v)
        return tangent_project(oracle.hess_vec(v)) + reg * v

    op = LinearOperator((n, n), matvec=matvec, dtype=float)
    try:
        d, info = cg(op, -g_tan, rtol=tol, atol=0.0, maxiter=max_cg_iter)
    except TypeError:
        d, info = cg(op, -g_tan, tol=tol, maxiter=max_cg_iter)
    d = tangent_project(d)

    if info != 0 or not np.all(np.isfinite(d)) or g_tan @ d >= 0:
        d = -g_tan
    return d


def load_returns_from_pickle(path, top_n=400):
    """Load the pickle file and build a daily return matrix for MVSK optimization."""
    with open(path, 'rb') as f:
        df = pickle.load(f)

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f'Expected a pandas.DataFrame, got {type(df)}')

    required = {'ts_code', 'trade_date', 'close', 'total_mv'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    # Keep the most liquid assets by average market cap to keep the optimization tractable.
    close_df = df[['ts_code', 'trade_date', 'close', 'total_mv']].copy()
    avg_mv = close_df.groupby('ts_code', as_index=False)['total_mv'].mean().sort_values('total_mv', ascending=False)
    selected_codes = avg_mv['ts_code'].head(top_n).tolist() if top_n is not None else avg_mv['ts_code'].tolist()
    close_df = close_df[close_df['ts_code'].isin(selected_codes)].copy()

    # Pivot to date x asset matrix using the latest available close price.
    wide = close_df.pivot_table(index='trade_date', columns='ts_code', values='close', aggfunc='last').sort_index()
    ret = wide.pct_change().dropna()
    ret = ret.loc[:, ret.std(axis=0) > 1e-12]
    return ret.to_numpy(dtype=float), list(ret.columns), list(ret.index)


def load_return_panel(path, top_n=None):
    """Load a wide or long return/price panel for real-data target sweeps.

    Supported inputs: pkl, csv, parquet. Long panels should contain an asset code
    column, a datetime/date column, and either `ret`/`return` or `close`.
    Wide panels are interpreted as already containing returns by asset.
    """
    path = str(path)
    if path.endswith(('.pkl', '.pickle')):
        with open(path, 'rb') as f:
            data = pickle.load(f)
    elif path.endswith('.csv'):
        data = pd.read_csv(path)
    elif path.endswith('.parquet'):
        data = pd.read_parquet(path)
    else:
        raise ValueError('Unsupported panel format. Use pkl, csv, or parquet.')

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f'Expected DataFrame panel, got {type(data)}')

    df = data.copy()
    lower = {str(c).lower(): c for c in df.columns}
    code_col = next((lower[k] for k in ['ts_code', 'symbol', 'asset', 'code', 'ticker'] if k in lower), None)
    time_col = next((lower[k] for k in ['datetime', 'time', 'trade_time', 'date', 'trade_date'] if k in lower), None)
    ret_col = next((lower[k] for k in ['ret', 'return', 'returns', 'pct_chg'] if k in lower), None)
    close_col = lower.get('close')

    if code_col and time_col and (ret_col or close_col):
        if ret_col:
            long = df[[code_col, time_col, ret_col]].copy()
            long[ret_col] = pd.to_numeric(long[ret_col], errors='coerce')
            if str(ret_col).lower() == 'pct_chg':
                long[ret_col] = long[ret_col] / 100.0
            wide = long.pivot_table(index=time_col, columns=code_col, values=ret_col, aggfunc='last')
        else:
            long = df[[code_col, time_col, close_col]].copy()
            long[close_col] = pd.to_numeric(long[close_col], errors='coerce')
            price = long.pivot_table(index=time_col, columns=code_col, values=close_col, aggfunc='last').sort_index()
            wide = price.pct_change()
    else:
        wide = df.copy()
        if time_col:
            wide = wide.set_index(time_col)

    wide = wide.sort_index()
    wide = wide.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    wide = wide.loc[:, wide.std(axis=0) > 1e-12]
    if top_n is not None and wide.shape[1] > top_n:
        vol_proxy = wide.abs().mean(axis=0).sort_values(ascending=False)
        wide = wide[vol_proxy.head(top_n).index]
    return wide.to_numpy(dtype=float), list(wide.columns), list(wide.index)


def split_time_series(R, dates, train_ratio=0.70):
    """Split a chronological return matrix into train / test sets with a 7:3 split."""
    R = np.asarray(R, dtype=float)
    dates = list(dates)
    if R.ndim != 2:
        raise ValueError('R must be a 2-D return matrix')
    if len(dates) != R.shape[0]:
        raise ValueError('Number of dates must match the number of return rows')
    if len(dates) < 3:
        raise ValueError('Need at least three time points for train/validation/test split')

    total = len(dates)
    n_train = int(np.floor(total * train_ratio))
    n_test = total - n_train
    if n_test <= 0:
        n_train = max(1, total - 1)
        n_test = total - n_train

    train_slice = slice(0, n_train)
    test_slice = slice(n_train, total)

    return {
        'train': (train_slice, R[train_slice]),
        'test': (test_slice, R[test_slice]),
        'dates': {
            'train': dates[train_slice],
            'test': dates[test_slice],
        },
        'sizes': {'train': n_train, 'test': n_test},
    }


def split_by_date(R, dates, split_date='2024-01-01'):
    """Chronological train/test split by timestamp/date label."""
    idx = pd.to_datetime(pd.Index(dates).astype(str), errors='coerce')
    if idx.isna().all():
        return split_time_series(R, dates, train_ratio=0.70)
    mask = idx < pd.Timestamp(split_date)
    if mask.sum() == 0 or (~mask).sum() == 0:
        return split_time_series(R, dates, train_ratio=0.70)
    train_slice = np.where(mask)[0]
    test_slice = np.where(~mask)[0]
    return {
        'train': (train_slice, R[train_slice]),
        'test': (test_slice, R[test_slice]),
        'dates': {
            'train': [dates[i] for i in train_slice],
            'test': [dates[i] for i in test_slice],
        },
        'sizes': {'train': len(train_slice), 'test': len(test_slice)},
    }


def generate_crra_conditioning_benchmark(T=2000, n=1000, kappa=1000, seed=7, c1=1.0, c2=3.0, c3=7.0, c4=14.0):
    """Generate a sample-rich CRRA conditioning benchmark with prescribed positive singular-spectrum conditioning."""
    rng = np.random.default_rng(seed)
    if T <= n:
        raise ValueError('The sample-rich CRRA benchmark requires T > n')

    # Build Q in the sample-time subspace orthogonal to 1_T.
    Zq = rng.standard_normal((T, n - 1))
    Zq = Zq - Zq.mean(axis=0, keepdims=True)
    Q, _ = np.linalg.qr(Zq, mode='reduced')

    # U spans the simplex tangent space, so A @ (1/n)1 = 0 and AU has the
    # requested positive singular spectrum.
    U = simplex_tangent_basis(n)
    singular_values = np.geomspace(1.0, float(kappa), n - 1)
    A_cond = (Q * singular_values) @ U.T

    # Add a nonzero mean vector so the equal-weight start is not a trivial
    # stationary point. The oracle will recover A_cond after column-centering.
    mu = 1e-3 * rng.standard_normal(n)
    R_cond = A_cond + mu[None, :]

    x0 = np.full(n, 1.0 / n, dtype=float)
    return {
        'R': R_cond,
        'x0': x0,
        'T': T,
        'n': n,
        'kappa': kappa,
        'seed': seed,
        'c': (c1, c2, c3, c4),
    }


def projected_simplex_residual(R, x, c1=0.2, c2=1.0, c3=0.2, c4=0.05):
    """Projected-gradient KKT residual for the simplex-constrained MVSK problem."""
    oracle = MVSKOracle(R, c1=c1, c2=c2, c3=c3, c4=c4)
    g = oracle.grad(x)
    return float(np.linalg.norm(x - project_to_simplex(x - g)))


def run_crra_conditioning_benchmark(kappas=(1, 10, 100, 1000), T=2000, n=1000, seed=7, max_iter=20, mode='large'):
    """Run the sample-rich CRRA conditioning benchmark and collect solver diagnostics."""
    results = []
    for kappa in kappas:
        bench = generate_crra_conditioning_benchmark(T=T, n=n, kappa=kappa, seed=seed)
        R = bench['R']
        start = time.perf_counter()
        x_star, f_star, g_star, history = optimize_mvsk(R, c1=bench['c'][0], c2=bench['c'][1], c3=bench['c'][2], c4=bench['c'][3], max_iter=max_iter, mode=mode)
        elapsed = time.perf_counter() - start
        metrics = compute_metrics(R, x_star)
        residual = projected_simplex_residual(R, x_star, c1=bench['c'][0], c2=bench['c'][1], c3=bench['c'][2], c4=bench['c'][3])
        results.append({
            'kappa_plus': kappa,
            'T': T,
            'n': n,
            'c1': bench['c'][0],
            'c2': bench['c'][1],
            'c3': bench['c'][2],
            'c4': bench['c'][3],
            'objective': float(f_star),
            'grad_norm': float(np.linalg.norm(g_star)),
            'projected_residual': residual,
            'runtime_sec': float(elapsed),
            'iterations': int(len(history)),
            'mean_return': float(metrics['mean_return']),
            'volatility': float(metrics['volatility']),
            'sharpe_proxy': float(metrics['sharpe_proxy']),
        })
    return pd.DataFrame(results), bench


def generate_synthetic_benchmark(T=252, n=100, seed=7, profiles=None):
    """Generate the paper-style synthetic MVSK benchmark from uniform returns."""
    if profiles is None:
        profiles = [
            ('return_seeking', (10.0, 1.0, 10.0, 1.0)),
            ('risk_averse', (1.0, 10.0, 1.0, 10.0)),
            ('balanced', (10.0, 10.0, 10.0, 10.0)),
        ]

    rng = np.random.default_rng(seed)
    R = rng.uniform(-0.1, 0.4, size=(T, n))
    mu = R.mean(axis=0)
    A = R - np.ones((T, 1)) * mu[None, :]
    x0 = np.full(n, 1.0 / n, dtype=float)

    return {
        'R': R,
        'mu': mu,
        'A': A,
        'x0': x0,
        'profiles': profiles,
        'T': T,
        'n': n,
        'seed': seed,
    }


def run_synthetic_benchmark(T=252, n=100, seed=7, max_iter=40, mode='auto', methods=('yand',)):
    """Run the YAND solver on the paper-style synthetic benchmark and report summary metrics."""
    bench = generate_synthetic_benchmark(T=T, n=n, seed=seed)
    R = bench['R']
    results = []

    for method in methods:
        for name, (c1, c2, c3, c4) in bench['profiles']:
            start = time.perf_counter()
            x_star, f_star, g_star, history = optimize_with_method(R, method=method, c1=c1, c2=c2, c3=c3, c4=c4, max_iter=max_iter, mode=mode)
            elapsed = time.perf_counter() - start
            metrics = compute_metrics(R, x_star)
            residual = projected_simplex_residual(R, x_star, c1=c1, c2=c2, c3=c3, c4=c4)
            results.append({
                'method': method,
                'profile': name,
                'c1': c1,
                'c2': c2,
                'c3': c3,
                'c4': c4,
                'objective': float(f_star),
                'grad_norm': float(np.linalg.norm(g_star)),
                'projected_residual': residual,
                'runtime_sec': float(elapsed),
                'iterations': int(len(history)),
                'mean_return': float(metrics['mean_return']),
                'volatility': float(metrics['volatility']),
                'sharpe_proxy': float(metrics['sharpe_proxy']),
                'max_weight': float(metrics['max_weight']),
                'num_nonzero': int(metrics['num_nonzero']),
            })

    return pd.DataFrame(results), bench


def run_paper_synthetic_grid(dims=(4, 10, 20, 40, 60, 80, 100, 120, 200, 400, 800), T=252, seed=7, max_iter=100, direct_threshold=100, methods=('yand', 'q-mvsk', 'ubdca', 'udca')):
    """Run the paper-style synthetic coefficient-stress grid for this implementation.

    This reproduces the benchmark instance family and reports YAND-style solver
    diagnostics. It does not include external baselines such as Q-MVSK, UBDCA, or
    UDCA unless those implementations are separately added.
    """
    rows = []
    for n in dims:
        mode = 'direct' if n <= direct_threshold else 'large'
        active_methods = list(methods)
        if n > 300:
            active_methods = [m for m in active_methods if m not in {'q-mvsk'}]
        summary, _ = run_synthetic_benchmark(T=T, n=n, seed=seed, max_iter=max_iter, mode=mode, methods=active_methods)
        summary.insert(0, 'n', n)
        summary.insert(1, 'T', T)
        summary.insert(2, 'mode', mode)
        rows.append(summary)
    detail = pd.concat(rows, ignore_index=True)
    best = detail.groupby(['n', 'profile'])['objective'].transform('min')
    detail['gap_to_best'] = detail['objective'] - best
    detail['objective_tie'] = detail['gap_to_best'].abs() <= 1e-6
    return detail


def normalize_code(code):
    """Normalize a stock code to the 6-digit format used by akshare."""
    if isinstance(code, str) and '.' in code:
        return code.split('.')[0]
    return str(code).zfill(6)


def get_stock_name_map():
    """Fetch the Chinese stock code -> name mapping from akshare."""
    import akshare as ak

    df = ak.stock_info_a_code_name()
    df = df.rename(columns={'code': 'code6', 'name': 'name_zh'})
    df['code6'] = df['code6'].astype(str).str.zfill(6)
    return df[['code6', 'name_zh']]


def compute_metrics(R, x):
    """Compute simple portfolio summary statistics for the solved weights."""
    R = np.asarray(R, dtype=float)
    x = np.asarray(x, dtype=float)
    mean_ret = R.mean(axis=0) @ x
    vol = np.sqrt((R @ x).var())
    skew = ((R @ x - mean_ret) ** 3).mean() / (vol ** 3 + 1e-12)
    kurt = ((R @ x - mean_ret) ** 4).mean() / (vol ** 4 + 1e-12)
    return {
        'mean_return': float(mean_ret),
        'volatility': float(vol),
        'skewness': float(skew),
        'kurtosis': float(kurt),
        'sharpe_proxy': float(mean_ret / (vol + 1e-12)),
        'max_weight': float(np.max(x)),
        'num_nonzero': int(np.sum(x > 1e-6)),
    }


def compute_real_metrics(R_test, x, market=None, periods_per_year=11712, cvar_alpha=0.01):
    """Annualized return/risk metrics used by the real-data target sweep."""
    r = np.asarray(R_test, dtype=float) @ np.asarray(x, dtype=float)
    mean_period = float(np.mean(r))
    vol_period = float(np.std(r))
    ann_return = (1.0 + mean_period) ** periods_per_year - 1.0
    ann_vol = vol_period * np.sqrt(periods_per_year)
    sharpe = ann_return / (ann_vol + 1e-12)
    q = np.quantile(r, cvar_alpha)
    cvar = float(np.mean(r[r <= q])) if np.any(r <= q) else float(q)
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    mdd = float(np.max(1.0 - wealth / (peak + 1e-12)))
    out = {
        'ann_return': float(ann_return),
        'ann_volatility': float(ann_vol),
        'sharpe': float(sharpe),
        'cvar1': cvar,
        'max_drawdown': mdd,
    }
    if market is not None:
        market = np.asarray(market, dtype=float)
        qs = np.quantile(market, np.linspace(0, 1, 11))
        excess = r
        worst = market <= qs[1]
        best = market >= qs[-2]
        middle = (~worst) & (~best)
        out['worst_decile_bps'] = float(np.mean(excess[worst]) * 1e4) if np.any(worst) else np.nan
        out['middle_deciles_bps'] = float(np.mean(excess[middle]) * 1e4) if np.any(middle) else np.nan
        out['best_decile_bps'] = float(np.mean(excess[best]) * 1e4) if np.any(best) else np.nan
    return out


def _solve_equality_qp(Sigma, mu, target_return=None, active=None):
    """Dense equality-constrained minimum-variance solution on active assets."""
    n = Sigma.shape[0]
    if active is None:
        active = np.ones(n, dtype=bool)
    idx = np.flatnonzero(active)
    k = idx.size
    if k == 0:
        raise np.linalg.LinAlgError('empty active set')

    S = Sigma[np.ix_(idx, idx)]
    ones = np.ones(k)
    if target_return is None:
        C = ones[:, None]
        b = np.array([1.0])
    else:
        C = np.column_stack((ones, mu[idx]))
        b = np.array([1.0, float(target_return)])

    jitter = 0.0
    for _ in range(5):
        try:
            SinvC = np.linalg.solve(S + jitter * np.eye(k), C)
            gram = C.T @ SinvC
            coeff = np.linalg.solve(gram, b)
            x_active = SinvC @ coeff
            x = np.zeros(n, dtype=float)
            x[idx] = x_active
            return x
        except np.linalg.LinAlgError:
            jitter = 1e-10 if jitter == 0.0 else jitter * 10.0

    SinvC = np.linalg.lstsq(S + jitter * np.eye(k), C, rcond=None)[0]
    coeff = np.linalg.lstsq(C.T @ SinvC, b, rcond=None)[0]
    x = np.zeros(n, dtype=float)
    x[idx] = SinvC @ coeff
    return x


def _long_only_equality_active_set(Sigma, mu, target_return=None, tol=1e-10, max_iter=None):
    """Active-set solve for long-only minimum variance with equality constraints."""
    n = Sigma.shape[0]
    if max_iter is None:
        max_iter = 3 * n + 10

    active = np.ones(n, dtype=bool)
    for _ in range(max_iter):
        x = _solve_equality_qp(Sigma, mu, target_return=target_return, active=active)
        active_idx = np.flatnonzero(active)
        if np.any(x[active] < -tol):
            active[active_idx[np.argmin(x[active])]] = False
            if active.sum() == 0:
                break
            continue

        grad = 2.0 * (Sigma @ x)
        if target_return is None:
            C_active = np.ones((active_idx.size, 1))
            C_all = np.ones((n, 1))
        else:
            C_active = np.column_stack((np.ones(active_idx.size), mu[active_idx]))
            C_all = np.column_stack((np.ones(n), mu))
        lam = np.linalg.lstsq(C_active, -grad[active], rcond=None)[0]
        reduced_grad = grad + C_all @ lam
        inactive = ~active
        if not np.any(inactive) or np.min(reduced_grad[inactive]) >= -1e-8:
            x[x < 0.0] = 0.0
            s = float(x.sum())
            if s > 0.0:
                x /= s
            return x

        inactive_idx = np.flatnonzero(inactive)
        active[inactive_idx[np.argmin(reduced_grad[inactive])]] = True

    raise np.linalg.LinAlgError('active-set equality solver did not converge')


def _long_only_markowitz_active_set(Sigma, mu, target_return, tol=1e-10, max_iter=None):
    """Fast active-set solver for long-only target mean-variance portfolios."""
    target = float(target_return)
    max_mu = float(np.max(mu))
    if target > max_mu + 1e-12:
        x = np.zeros(Sigma.shape[0], dtype=float)
        x[int(np.argmax(mu))] = 1.0
        return x, False, 'target_return is above the long-only feasible range'

    x = _long_only_equality_active_set(Sigma, mu, tol=tol, max_iter=max_iter)
    if float(mu @ x) >= target - 1e-9:
        return x, True, 'active-set minimum-variance solution'

    x = _long_only_equality_active_set(Sigma, mu, target_return=target, tol=tol, max_iter=max_iter)
    return x, True, 'active-set target-return solution'


def _mv_problem_stats(R_train):
    R_train = np.asarray(R_train, dtype=float)
    mu = R_train.mean(axis=0)
    A = R_train - mu[None, :]
    Sigma = (A.T @ A) / max(1, R_train.shape[0])
    Sigma = 0.5 * (Sigma + Sigma.T) + 1e-10 * np.eye(R_train.shape[1])
    return mu, Sigma


def _solve_mv_target_from_stats(mu, Sigma, target_return):
    n = len(mu)
    try:
        x, success, message = _long_only_markowitz_active_set(Sigma, mu, target_return)
        res = OptimizeResult(
            x=x,
            success=bool(success),
            message=message,
            fun=float(x @ Sigma @ x),
            nit=0,
        )
        return x, res
    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
        pass

    x0 = np.full(n, 1.0 / n)
    cons = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 'jac': lambda x: np.ones(n)},
        {'type': 'ineq', 'fun': lambda x: mu @ x - target_return, 'jac': lambda x: mu},
    )
    bounds = [(0.0, 1.0)] * n
    res = minimize(
        lambda x: float(x @ Sigma @ x),
        x0,
        jac=lambda x: 2.0 * Sigma @ x,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 500, 'ftol': 1e-12, 'disp': False},
    )
    return project_to_simplex(res.x), res


def solve_mv_target(R_train, target_return):
    """Exact long-only mean-variance target portfolio.

    The common case is handled by a dense active-set Markowitz solve, which is
    much faster than repeatedly invoking SLSQP in target sweeps. SLSQP remains
    as a robustness fallback for pathological numerical cases.
    """
    mu, Sigma = _mv_problem_stats(R_train)
    return _solve_mv_target_from_stats(mu, Sigma, target_return)


_MVSK_WORKER_CONTEXT = None


def _init_mvsk_worker(R_train, mu, base_c2, base_c3, base_c4, method, mode, max_iter):
    global _MVSK_WORKER_CONTEXT
    _MVSK_WORKER_CONTEXT = (R_train, mu, base_c2, base_c3, base_c4, method, mode, max_iter)


def _mvsk_candidate_worker(c1):
    R_train, mu, base_c2, base_c3, base_c4, method, mode, max_iter = _MVSK_WORKER_CONTEXT
    coeffs = (float(c1), base_c2, base_c3, base_c4)
    x, f, g, hist = optimize_with_method(
        R_train,
        method=method,
        c1=coeffs[0],
        c2=coeffs[1],
        c3=coeffs[2],
        c4=coeffs[3],
        max_iter=max_iter,
        mode=mode,
    )
    return (float(mu @ x), x, f, g, hist, coeffs)


def _mvsk_target_candidates(R_train, profile='kurtosis', method='yand', mode='large', max_iter=100, c1_grid=None, n_jobs=1):
    """Precompute reusable MVSK candidates for target-return sweeps."""
    R_train = np.asarray(R_train, dtype=float)
    mu = R_train.mean(axis=0)
    if c1_grid is None:
        c1_grid = np.geomspace(1e-4, 1e3, 10)
    _, base_c2, base_c3, base_c4 = normalized_mvsk_coefficients(R_train, profile=profile, c1=1.0)

    c1_grid = [float(c1) for c1 in c1_grid]
    n_jobs = int(n_jobs or 1)
    _init_mvsk_worker(R_train, mu, base_c2, base_c3, base_c4, method, mode, max_iter)
    if n_jobs <= 1 or len(c1_grid) <= 1:
        return [_mvsk_candidate_worker(c1) for c1 in c1_grid]

    if n_jobs < 0:
        n_jobs = os.cpu_count() or 1
    workers = min(n_jobs, len(c1_grid))
    context = mp.get_context('fork') if hasattr(os, 'fork') else None
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_mvsk_worker,
        initargs=(R_train, mu, base_c2, base_c3, base_c4, method, mode, max_iter),
    ) as pool:
        return list(tqdm(pool.map(_mvsk_candidate_worker, c1_grid), total=len(c1_grid), desc='MVSK c1 grid'))


def _select_mvsk_candidate(candidates, target_return):
    best = min(candidates, key=lambda cand: abs(cand[0] - target_return))
    ret, x, f, g, hist, coeffs = best
    return x, f, g, hist, coeffs, ret


def normalized_mvsk_coefficients(R_train, profile='kurtosis', c1=1.0):
    """Build profile coefficients normalized by equal-weight moments."""
    R_train = np.asarray(R_train, dtype=float)
    n = R_train.shape[1]
    xeq = np.full(n, 1.0 / n)
    oracle = MVSKOracle(R_train, c1=1.0, c2=1.0, c3=1.0, c4=1.0)
    z = oracle.A @ xeq
    m2 = max(float(np.mean(z**2)), 1e-12)
    m3 = max(abs(float(np.mean(z**3))), 1e-12)
    m4 = max(float(np.mean(z**4)), 1e-12)
    profile = profile.lower()
    if profile == 'skew':
        return (c1, 1.0 / m2, 1.0 / m3, 0.25 / m4)
    if profile == 'balanced':
        return (c1, 1.0 / m2, 1.0 / m3, 1.0 / m4)
    return (c1, 1.0 / m2, 0.25 / m3, 1.0 / m4)


def solve_mvsk_for_target(R_train, target_return, profile='kurtosis', method='yand', mode='large', max_iter=100, candidates=None, n_jobs=1):
    """Calibrate c1 so MVSK in-sample return is close to the MV return floor."""
    if candidates is None:
        candidates = _mvsk_target_candidates(R_train, profile=profile, method=method, mode=mode, max_iter=max_iter, n_jobs=n_jobs)
    return _select_mvsk_candidate(candidates, target_return)


def run_real_target_sweep(panel_path, q_values=(0.40, 0.50, 0.60), split_date='2024-01-01', top_n=None, profile='kurtosis', method='yand', mode='large', max_iter=100, periods_per_year=11712, n_jobs=1):
    """Run the paper-style MV-vs-MVSK real-data target sweep."""
    R, symbols, dates = load_return_panel(panel_path, top_n=top_n)
    split = split_by_date(R, dates, split_date=split_date)
    _, R_train = split['train']
    _, R_test = split['test']
    mu, Sigma = _mv_problem_stats(R_train)
    market = R_test.mean(axis=1)
    mvsk_candidates = _mvsk_target_candidates(R_train, profile=profile, method=method, mode=mode, max_iter=max_iter, n_jobs=n_jobs)

    rows = []
    for q in tqdm(q_values):
        target = float(np.quantile(mu, q))
        x_mv, mv_res = _solve_mv_target_from_stats(mu, Sigma, target)
        x_mvsk, f, g, hist, coeffs, mvsk_train_ret = solve_mvsk_for_target(
            R_train,
            target,
            profile=profile,
            method=method,
            mode=mode,
            max_iter=max_iter,
            candidates=mvsk_candidates,
            n_jobs=n_jobs,
        )

        mv_metrics = compute_real_metrics(R_test, x_mv, market=market, periods_per_year=periods_per_year)
        mvsk_metrics = compute_real_metrics(R_test, x_mvsk, market=market, periods_per_year=periods_per_year)
        active_share = 0.5 * float(np.sum(np.abs(x_mvsk - x_mv)))
        rows.append({
            'q': q,
            'target_return_period': target,
            'profile': profile,
            'method': method,
            'train_T': R_train.shape[0],
            'test_T': R_test.shape[0],
            'n': R_train.shape[1],
            'mv_train_return': float(mu @ x_mv),
            'mvsk_train_return': float(mvsk_train_ret),
            'mv_ann_return': mv_metrics['ann_return'],
            'mvsk_ann_return': mvsk_metrics['ann_return'],
            'delta_return_pp': 100.0 * (mvsk_metrics['ann_return'] - mv_metrics['ann_return']),
            'mv_sharpe': mv_metrics['sharpe'],
            'mvsk_sharpe': mvsk_metrics['sharpe'],
            'delta_sharpe': mvsk_metrics['sharpe'] - mv_metrics['sharpe'],
            'delta_cvar1_pp': 100.0 * (mvsk_metrics['cvar1'] - mv_metrics['cvar1']),
            'delta_mdd_pp': 100.0 * (mv_metrics['max_drawdown'] - mvsk_metrics['max_drawdown']),
            'active_share': active_share,
            'mvsk_objective': float(f),
            'mvsk_projected_residual': projected_simplex_residual(R_train, x_mvsk, c1=coeffs[0], c2=coeffs[1], c3=coeffs[2], c4=coeffs[3]),
            'mvsk_iterations': len(hist),
            'worst_decile_delta_bps': mvsk_metrics.get('worst_decile_bps', np.nan) - mv_metrics.get('worst_decile_bps', np.nan),
            'middle_deciles_delta_bps': mvsk_metrics.get('middle_deciles_bps', np.nan) - mv_metrics.get('middle_deciles_bps', np.nan),
            'best_decile_delta_bps': mvsk_metrics.get('best_decile_bps', np.nan) - mv_metrics.get('best_decile_bps', np.nan),
        })
    return pd.DataFrame(rows)


def tune_params_on_validation(R_train, R_val, param_grid, max_iter=100):
    """Select the best parameter set by validation Sharpe proxy."""
    best = None
    for params in param_grid:
        c1, c2, c3, c4 = params
        x_star, _, _, _ = optimize_mvsk(R_train, c1=c1, c2=c2, c3=c3, c4=c4, max_iter=max_iter)
        val_metrics = compute_metrics(R_val, x_star)
        score = val_metrics['sharpe_proxy']
        cand = {
            'params': params,
            'x_star': x_star,
            'score': score,
            'val_metrics': val_metrics,
        }
        if best is None or score > best['score']:
            best = cand
    return best


def solve_simplex_slsqp(objective, grad, x0, max_iter=200):
    """Solve a smooth simplex-constrained subproblem with SciPy SLSQP."""
    n = len(x0)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 'jac': lambda x: np.ones(n)},)
    bounds = [(0.0, 1.0)] * n
    res = minimize(
        objective,
        x0,
        method='SLSQP',
        jac=grad,
        bounds=bounds,
        constraints=cons,
        options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': False},
    )
    return project_to_simplex(res.x), res


def optimize_q_mvsk(R, c1=0.2, c2=1.0, c3=0.2, c4=0.05, max_iter=50, tol=1e-6, proximal=1e-4):
    """Q-MVSK-style quadratic majorization baseline.

    Each outer step builds a PSD quadratic surrogate from the local Hessian
    plus a spectral/proximal shift, then solves the simplex QP with SLSQP.
    This is a local reproducible baseline, not the original authors' code.
    """
    oracle = MVSKOracle(R, c1=c1, c2=c2, c3=c3, c4=c4)
    n = R.shape[1]
    x = np.full(n, 1.0 / n)
    history = []

    for k in range(max_iter):
        oracle.set_current_x(x)
        g = oracle.grad(x)
        residual = np.linalg.norm(x - project_to_simplex(x - g))
        history.append((k, oracle.value(x), residual, x.copy()))
        if residual < tol:
            break

        H = np.zeros((n, n), dtype=float)
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            H[:, i] = oracle.hess_vec(ei)
        H = 0.5 * (H + H.T)
        min_eig = float(np.linalg.eigvalsh(H).min())
        shift = max(proximal, -min_eig + proximal)
        Hq = H + shift * np.eye(n)
        q = g - Hq @ x

        def obj(y):
            return 0.5 * y @ Hq @ y + q @ y

        def jac(y):
            return Hq @ y + q

        x_new, _ = solve_simplex_slsqp(obj, jac, x, max_iter=200)
        if oracle.value(x_new) > oracle.value(x) and np.linalg.norm(x_new - x) > 1e-12:
            d = tangent_project(x_new - x)
            _, x_new = exact_quartic_line_search(oracle, x, d)
            x_new = project_to_simplex(x_new)
        if np.linalg.norm(x_new - x) < 1e-10:
            break
        x = x_new

    return x, oracle.value(x), oracle.grad(x), history


def optimize_udca(R, c1=0.2, c2=1.0, c3=0.2, c4=0.05, max_iter=200, tol=1e-6, step=1.0):
    """UDCA-style projected first-order DC baseline.

    This baseline uses projected descent with exact quartic line search. It is
    intentionally simple and reproducible for comparison on the same instances.
    """
    oracle = MVSKOracle(R, c1=c1, c2=c2, c3=c3, c4=c4)
    n = R.shape[1]
    x = np.full(n, 1.0 / n)
    history = []

    for k in range(max_iter):
        oracle.set_current_x(x)
        g = oracle.grad(x)
        residual = np.linalg.norm(x - project_to_simplex(x - g))
        history.append((k, oracle.value(x), residual, x.copy()))
        if residual < tol:
            break
        x_pg = project_to_simplex(x - step * g)
        d = tangent_project(x_pg - x)
        if np.linalg.norm(d) < 1e-12:
            d = -tangent_project(g)
        _, x_new = exact_quartic_line_search(oracle, x, d)
        x_new = project_to_simplex(x_new)
        if oracle.value(x_new) > oracle.value(x):
            x_new = project_to_simplex(x - min(step, 1.0) * g)
        if np.linalg.norm(x_new - x) < 1e-10:
            break
        x = x_new

    return x, oracle.value(x), oracle.grad(x), history


def optimize_ubdca(R, c1=0.2, c2=1.0, c3=0.2, c4=0.05, max_iter=200, tol=1e-6, step=1.0, beta=0.7):
    """UBDCA-style boosted projected baseline with monotone safeguard."""
    oracle = MVSKOracle(R, c1=c1, c2=c2, c3=c3, c4=c4)
    n = R.shape[1]
    x = np.full(n, 1.0 / n)
    x_prev = x.copy()
    history = []

    for k in range(max_iter):
        oracle.set_current_x(x)
        g = oracle.grad(x)
        residual = np.linalg.norm(x - project_to_simplex(x - g))
        history.append((k, oracle.value(x), residual, x.copy()))
        if residual < tol:
            break

        y = project_to_simplex(x + beta * (x - x_prev)) if k > 0 else x.copy()
        oracle.set_current_x(y)
        gy = oracle.grad(y)
        y_pg = project_to_simplex(y - step * gy)
        d = tangent_project(y_pg - y)
        if np.linalg.norm(d) < 1e-12:
            d = -tangent_project(gy)
        _, cand = exact_quartic_line_search(oracle, y, d)
        cand = project_to_simplex(cand)

        oracle.set_current_x(x)
        if oracle.value(cand) > oracle.value(x):
            cand = project_to_simplex(x - min(step, 1.0) * g)
        if oracle.value(cand) > oracle.value(x):
            cand = x.copy()

        if np.linalg.norm(cand - x) < 1e-10:
            break
        x_prev, x = x, cand

    oracle.set_current_x(x)
    return x, oracle.value(x), oracle.grad(x), history


def optimize_with_method(R, method='yand', c1=0.2, c2=1.0, c3=0.2, c4=0.05, max_iter=100, tol=1e-7, mode='auto'):
    """Dispatch optimizer by method name."""
    method = method.lower()
    if method in {'yand', 'yand-mvsk'}:
        return optimize_mvsk(R, c1=c1, c2=c2, c3=c3, c4=c4, max_iter=max_iter, tol=tol, mode=mode)
    if method in {'yand-householder', 'householder'}:
        return optimize_mvsk(R, c1=c1, c2=c2, c3=c3, c4=c4, max_iter=max_iter, tol=tol, mode='householder')
    if method in {'q-mvsk', 'qmvsk'}:
        return optimize_q_mvsk(R, c1=c1, c2=c2, c3=c3, c4=c4, max_iter=max_iter, tol=tol)
    if method == 'udca':
        return optimize_udca(R, c1=c1, c2=c2, c3=c3, c4=c4, max_iter=max_iter, tol=tol)
    if method == 'ubdca':
        return optimize_ubdca(R, c1=c1, c2=c2, c3=c3, c4=c4, max_iter=max_iter, tol=tol)
    raise ValueError(f'Unknown method: {method}')


def optimize_mvsk(R, c1=0.2, c2=1.0, c3=0.2, c4=0.05, max_iter=100, tol=1e-7, mode='auto'):
    oracle = MVSKOracle(R, c1=c1, c2=c2, c3=c3, c4=c4)
    n = R.shape[1]
    if mode == 'auto':
        mode = 'direct' if n <= 150 else 'large'
    U = simplex_tangent_basis(n) if mode in {'direct', 'householder'} else None

    x = np.full(n, 1.0 / n, dtype=float)
    x = project_to_simplex(x)

    history = []
    for k in range(max_iter):
        oracle.set_current_x(x)
        g = oracle.grad(x)
        residual = np.linalg.norm(x - project_to_simplex(x - g))
        history.append((k, oracle.value(x), residual, x.copy()))
        if residual < tol:
            break

        if mode == 'direct':
            d = affine_normal_direction(oracle, x, U)
        elif mode == 'householder':
            d = affine_normal_householder_direction(oracle, x, U)
        elif mode == 'large':
            d = matrix_free_tangent_direction(oracle, x)
        else:
            raise ValueError(f'Unknown optimizer mode: {mode}')

        if d is None or np.linalg.norm(d) < 1e-12:
            break
        d = tangent_project(d)

        alpha, x_new = exact_quartic_line_search(oracle, x, d)

        x_new = project_to_simplex(x_new)
        if not np.isfinite(oracle.value(x_new)):
            break

        # Accept only if it actually decreases the objective.
        if oracle.value(x_new) < oracle.value(x) - 1e-12:
            x = x_new
        else:
            # Fallback: a small step in the negative gradient direction.
            d0 = -tangent_project(g)
            alpha0, x0 = exact_quartic_line_search(oracle, x, d0)
            x = project_to_simplex(x0)

    final_value = oracle.value(x)
    final_grad = oracle.grad(x)
    return x, final_value, final_grad, history


def parse_args():
    parser = argparse.ArgumentParser(description='Run the YAND-style MVSK solver on real pickle data or the paper-style synthetic benchmark.')
    parser.add_argument('--pickle', default='daily_batch_20240101_20251231_2dc5b4f1d287.pkl', help='Path to the pickle file to load.')
    parser.add_argument('--top_n', type=int, default=400, help='Number of most liquid assets to keep for the run.')
    parser.add_argument('--max_iter', type=int, default=40, help='Maximum optimizer iterations.')
    parser.add_argument('--synthetic', action='store_true', help='Run the paper-style synthetic benchmark instead of real data.')
    parser.add_argument('--crra', action='store_true', help='Run the paper-style CRRA conditioning benchmark instead of real data.')
    parser.add_argument('--paper_synthetic_grid', action='store_true', help='Run the paper-style synthetic dimension grid for this implementation.')
    parser.add_argument('--real_target_sweep', action='store_true', help='Run the paper-style MV-vs-MVSK target sweep on a 5-minute return panel.')
    parser.add_argument('--panel', default=None, help='Path to a 5-minute A-share return/price panel for --real_target_sweep.')
    parser.add_argument('--real_top_n', type=int, default=0, help='Optional asset cap for --real_target_sweep; 0 keeps the full panel.')
    parser.add_argument('--synthetic_n', type=int, default=100, help='Asset dimension for the synthetic benchmark.')
    parser.add_argument('--synthetic_T', type=int, default=252, help='Sample length for the synthetic benchmark.')
    parser.add_argument('--synthetic_seed', type=int, default=7, help='Random seed for the synthetic benchmark.')
    parser.add_argument('--dims', default='4,10,20,40,60,80,100,120,200,400,800', help='Comma-separated dimensions for --paper_synthetic_grid.')
    parser.add_argument('--methods', default='yand,q-mvsk,ubdca,udca', help='Comma-separated methods for synthetic grid.')
    parser.add_argument('--q_values', default='0.40,0.50,0.60', help='Comma-separated target quantiles for --real_target_sweep.')
    parser.add_argument('--profile', choices=('skew', 'kurtosis', 'balanced'), default='kurtosis', help='MVSK profile for real-data target sweep.')
    parser.add_argument('--split_date', default='2024-01-01', help='Date split for --real_target_sweep.')
    parser.add_argument('--periods_per_year', type=int, default=11712, help='Annualization periods for 5-minute data.')
    parser.add_argument('--mode', choices=('auto', 'direct', 'large', 'householder'), default='auto', help='Optimizer mode.')
    parser.add_argument('--n_jobs', type=int, default=1, help='Parallel workers for independent MVSK c1-grid optimizations. Use 1 to disable parallelism.')
    parser.add_argument('--check_executable', default=None, help='Path to check; exits after reporting whether it is an executable regular file.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.crra:
        crra_T = 2000 if args.synthetic_T == 252 else args.synthetic_T
        crra_n = 1000 if args.synthetic_n == 100 else args.synthetic_n
        summary, bench = run_crra_conditioning_benchmark(kappas=(1, 10, 100, 1000), T=crra_T, n=crra_n, seed=args.synthetic_seed, max_iter=args.max_iter, mode=args.mode)
        summary_path = 'yand_crra_conditioning_summary.csv'
        summary.to_csv(summary_path, index=False)
        print(f'Generated CRRA conditioning benchmark: T={crra_T}, N={crra_n}, CRRA c=(1,3,7,14)')
        print('\n=== CRRA Conditioning Benchmark ===')
        print(summary.to_string(index=False))
        print('\nCSV exported to:', summary_path)
        raise SystemExit(0)

    if args.paper_synthetic_grid:
        dims = tuple(int(part.strip()) for part in args.dims.split(',') if part.strip())
        methods = tuple(part.strip() for part in args.methods.split(',') if part.strip())
        summary = run_paper_synthetic_grid(dims=dims, T=args.synthetic_T, seed=args.synthetic_seed, max_iter=args.max_iter, methods=methods)
        summary_path = 'yand_paper_synthetic_grid_summary.csv'
        summary.to_csv(summary_path, index=False)
        print(f'Generated paper-style synthetic grid: T={args.synthetic_T}, dims={dims}, seed={args.synthetic_seed}')
        print('\n=== Paper-Style Synthetic Grid ===')
        print(summary.to_string(index=False))
        print('\nCSV exported to:', summary_path)
        raise SystemExit(0)

    if args.synthetic:
        methods = tuple(part.strip() for part in args.methods.split(',') if part.strip())
        summary, bench = run_synthetic_benchmark(T=args.synthetic_T, n=args.synthetic_n, seed=args.synthetic_seed, max_iter=args.max_iter, mode=args.mode, methods=methods)
        summary_path = 'yand_synthetic_benchmark_summary.csv'
        summary.to_csv(summary_path, index=False)
        print(f'Generated synthetic benchmark: T={bench["T"]}, N={bench["n"]}, seed={bench["seed"]}')
        print('\n=== Synthetic Benchmark (paper-style) ===')
        print(summary.to_string(index=False))
        print('\nCSV exported to:', summary_path)
        print('\nSynthetic benchmark profiles follow the paper:')
        for name, coeffs in bench['profiles']:
            print('  -', name, 'c=', coeffs)
        raise SystemExit(0)

    if args.real_target_sweep:
        if not args.panel:
            raise SystemExit('Please provide --panel path to a 5-minute return/price panel for --real_target_sweep.')
        q_values = tuple(float(part.strip()) for part in args.q_values.split(',') if part.strip())
        summary = run_real_target_sweep(
            args.panel,
            q_values=q_values,
            split_date=args.split_date,
            top_n=(args.real_top_n if args.real_top_n > 0 else None),
            profile=args.profile,
            method='yand',
            mode=args.mode,
            max_iter=args.max_iter,
            periods_per_year=args.periods_per_year,
            n_jobs=args.n_jobs,
        )
        summary_path = 'yand_real_target_sweep_summary.csv'
        summary.to_csv(summary_path, index=False)
        print(f'Generated real-data target sweep: panel={args.panel}, q={q_values}, profile={args.profile}')
        print('\n=== Real Target Sweep ===')
        print(summary.to_string(index=False))
        print('\nCSV exported to:', summary_path)
        raise SystemExit(0)

    try:
        R, symbols, dates = load_returns_from_pickle(args.pickle, top_n=args.top_n)
        print(f'Loaded real returns matrix: T={R.shape[0]}, N={R.shape[1]}')
        print(f'First date: {dates[0]}, last date: {dates[-1]}')
        print(f'First assets: {symbols[:10]}')

        split = split_time_series(R, dates, train_ratio=0.70)
        train_slice, R_train = split['train']
        test_slice, R_test = split['test']

        x_star, f_star, g_star, history = optimize_mvsk(R_train, c1=0.15, c2=1.0, c3=0.30, c4=0.08, max_iter=args.max_iter, mode=args.mode)
        metrics_train = compute_metrics(R_train, x_star)
        metrics_test = compute_metrics(R_test, x_star)

        equal_weight = np.full(R_test.shape[1], 1.0 / R_test.shape[1], dtype=float)
        metrics_equal = compute_metrics(R_test, equal_weight)

        print('\n=== Time Split Summary (7:3) ===')
        print('Train dates:', split['dates']['train'][0], '->', split['dates']['train'][-1], 'T=', split['sizes']['train'])
        print('Test dates: ', split['dates']['test'][0], '->', split['dates']['test'][-1], 'T=', split['sizes']['test'])
        print('Training objective:', round(f_star, 8))
        print('Training gradient norm:', round(np.linalg.norm(g_star), 8))
        print('Training projected residual:', round(projected_simplex_residual(R_train, x_star, c1=0.15, c2=1.0, c3=0.30, c4=0.08), 8))
        print('Test Sharpe proxy:', round(metrics_test['sharpe_proxy'], 6))
        print('Test mean return:', round(metrics_test['mean_return'], 8))
        print('Test volatility:', round(metrics_test['volatility'], 8))
        print('Test vs equal-weight Sharpe delta:', round(metrics_test['sharpe_proxy'] - metrics_equal['sharpe_proxy'], 6))
    except Exception as exc:
        print('Falling back to synthetic demo because real data loading failed:', exc)
        rng = np.random.default_rng(7)
        T = 300
        n = 40
        factors = rng.standard_normal((T, 4))
        loadings = rng.standard_normal((4, n))
        noise = 0.15 * rng.standard_normal((T, n))
        R = factors @ loadings + noise
        dates = list(range(T))
        split = split_time_series(R, dates, train_ratio=0.70)
        _, R_train = split['train']
        _, R_test = split['test']
        x_star, f_star, g_star, history = optimize_mvsk(R_train, c1=0.15, c2=1.0, c3=0.30, c4=0.08, max_iter=args.max_iter, mode=args.mode)
        metrics_train = compute_metrics(R_train, x_star)
        metrics_test = compute_metrics(R_test, x_star)
        equal_weight = np.full(R_test.shape[1], 1.0 / R_test.shape[1], dtype=float)
        metrics_equal = compute_metrics(R_test, equal_weight)

        print('\n=== Time Split Summary (7:3) ===')
        print('Train T=', split['sizes']['train'], 'Test T=', split['sizes']['test'])
        print('Training objective:', round(f_star, 8))
        print('Training gradient norm:', round(np.linalg.norm(g_star), 8))
        print('Test Sharpe proxy:', round(metrics_test['sharpe_proxy'], 6))
        print('Test mean return:', round(metrics_test['mean_return'], 8))
        print('Test volatility:', round(metrics_test['volatility'], 8))
        print('Test vs equal-weight Sharpe delta:', round(metrics_test['sharpe_proxy'] - metrics_equal['sharpe_proxy'], 6))

    metrics = compute_metrics(R_test, x_star)

    print('\n=== YAND-MVSK Result Summary ===')
    print('Objective:', f_star)
    print('Gradient norm:', np.linalg.norm(g_star))
    print('Iterations performed:', len(history))
    print('Train mean return:', round(metrics_train['mean_return'], 6))
    print('Train volatility:', round(metrics_train['volatility'], 6))
    print('Test mean return:', round(metrics['mean_return'], 6))
    print('Test volatility:', round(metrics['volatility'], 6))
    print('Test skewness:', round(metrics['skewness'], 6))
    print('Test kurtosis:', round(metrics['kurtosis'], 6))
    print('Test Sharpe proxy:', round(metrics['sharpe_proxy'], 6))
    print('Max weight:', round(x_star.max(), 6))
    print('Nonzero weights:', int(np.sum(x_star > 1e-6)))

    summary = pd.DataFrame([
        {'split': 'train', 'T': R_train.shape[0], 'mean_return': metrics_train['mean_return'], 'volatility': metrics_train['volatility'], 'sharpe_proxy': metrics_train['sharpe_proxy']},
        {'split': 'test', 'T': R_test.shape[0], 'mean_return': metrics_test['mean_return'], 'volatility': metrics_test['volatility'], 'sharpe_proxy': metrics_test['sharpe_proxy']},
        {'split': 'equal_weight_test', 'T': R_test.shape[0], 'mean_return': metrics_equal['mean_return'], 'volatility': metrics_equal['volatility'], 'sharpe_proxy': metrics_equal['sharpe_proxy']},
    ])
    summary.to_csv('yand_mvsk_time_split_summary.csv', index=False)
    print('\nCSV exported to: yand_mvsk_time_split_summary.csv')

    print('\n=== Portfolio Weights Table ===')
    if 'symbols' in locals():
        table = pd.DataFrame({'asset': symbols[:len(x_star)], 'weight': x_star}).sort_values('weight', ascending=False)
        top100 = table.head(100).reset_index(drop=True)
        top100['weight'] = top100['weight'].round(8)
        print(top100.to_string(index=False))
        print('\n... top 100 shown; total assets =', len(x_star))

        out_path = 'yand_mvsk_weights.csv'
        table.to_csv(out_path, index=False)
        print('\nCSV exported to:', out_path)

        name_map = get_stock_name_map()
        mapping = pd.DataFrame({
            'asset': table['asset'],
            'code6': table['asset'].apply(normalize_code),
            'name_zh': table['asset'].apply(normalize_code).map(dict(zip(name_map['code6'], name_map['name_zh'])))
        }).drop_duplicates(subset=['asset']).sort_values('asset')
        mapping_path = 'yand_mvsk_stock_name_map.csv'
        mapping.to_csv(mapping_path, index=False)
        print('Chinese name mapping exported to:', mapping_path)
        print('\n=== Stock Code -> Chinese Name Mapping (sample) ===')
        print(mapping.head(30).to_string(index=False))
    else:
        print('Asset names unavailable in synthetic demo mode.')
        table = pd.DataFrame({'weight': np.round(x_star, 6)})
        print(table.to_string())

# venv_sys/bin/python yand_mvsk_solver.py

# (.venv) zhouyue@zhouyuedeMacBook-Pro YAND %  /Users/zhouyue/Projects/YAND/.venv_sys/bin/python yand_mvsk_solver.py --top_n 120 --max_iter 20
# Loaded real returns matrix: T=74, N=120
