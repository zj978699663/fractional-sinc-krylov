import csv
from statistics import median

import numpy as np
from scipy import fft, linalg, sparse, special
from threadpoolctl import threadpool_limits

from common import (
    BUILD,
    DATA,
    U64,
    _divide_up,
    _next_down_positive,
    add_up,
    basis,
    frobenius_upper,
    gamma_p,
    interval_upper,
    iv,
    ivfloat,
    krylov,
    multiply_up,
    normA,
    prod_radius,
    rounded,
    sinc_bound,
    sinc_rule,
    sub_radius,
    write_table,
)


NS = (1024, 4096, 16384)
KS = (8, 16, 24, 32, 40, 48)
EPS, DELTA, TAU = 0.4, 0.01, 1e-3
METHODS = ("direct", "finite-vector")


def main():
    BUILD.mkdir(exist_ok=True)
    times = timings()
    rows = []
    with threadpool_limits(limits=1):
        for n in NS:
            dims = sparse_sequences(n)
            for rec in (0, 1):
                total = [
                    sum(
                        times[rep, n, stage, rec, "finite-vector"]
                        for stage in (1, 2, 3)
                    )
                    for rep in range(3)
                ]
                ks = "(" + ",".join(map(str, dims[rec])) + ")"
                rows.append(
                    [
                        n,
                        4 * rec,
                        ks,
                        *(rounded(t) for t in (median(total), min(total), max(total))),
                    ]
                )
    write_table(
        5, ["n", "k", "stopping_dimensions", "median_s", "min_s", "max_s"], rows
    )


def timings():
    times = {}
    with (DATA / "timings.csv").open(encoding="utf-8", newline="") as stream:
        for r in csv.DictReader(stream):
            key = (
                int(r["repetition"]),
                int(r["n"]),
                int(r["stage"]),
                int(r["recycling"]),
                r["validator"],
            )
            assert key not in times and float(r["total_s"]) > 0
            times[key] = float(r["total_s"])
    assert times.keys() == {
        (rep, n, stage, rec, method)
        for rep in range(3)
        for n in NS
        for stage in (1, 2, 3)
        for rec in (0, 1)
        for method in METHODS
    }
    return times


def sparse_sequences(n):
    rng = np.random.default_rng(20261011 + n)
    b0 = rng.normal(size=n)
    b0 /= np.linalg.norm(b0)
    U = np.empty((n, 0))
    dims = [[], []]
    for stage in range(3):
        diag = 2 + (0.05 + 0.003 * stage)
        off = -np.ones(n - 1)
        A = sparse.diags([off, np.full(n, diag), off], [-1, 0, 1], format="csr")
        b = b0 + 0.03 * stage * rng.normal(size=n) / np.sqrt(n)
        b /= np.linalg.norm(b)
        S = rng.normal(size=(96, n)) / np.sqrt(96)
        K = krylov(A, b, KS[-1])
        ell = float(np.nextafter(diag - 2, -np.inf))
        u = float(np.nextafter(diag + 2, np.inf))
        rule = sinc_rule(0.5, ell, u, 1.0, TAU)
        rule["eta_sinc"] = sinc_bound(rule, 0.5, ell, u, frobenius_upper(b))
        assert rule["nodes"] == 26
        nu = int(np.ceil(4 / EPS**2 * np.log(len(KS) * rule["nodes"] / DELTA)))
        assert nu == 242
        O = np.random.default_rng(20261012 + n + stage).normal(size=(nu, n)) / np.sqrt(
            nu
        )

        eigen = diag - 2 * np.cos(np.pi * np.arange(1, n + 1) / (n + 1))
        x_ref = fft.idst(
            fft.dst(b, type=1, norm="ortho") / np.sqrt(eigen), type=1, norm="ortho"
        )
        low = resolvent(diag, b, 128)
        high = resolvent(diag, b, 256)
        gap = max(normA(A, x_ref - high), normA(A, low - high))
        for rec in (0, 1):
            k, V, error = scan(
                A, b, K, U if rec else K[:, :0], S, O, rule, ell, u, x_ref
            )
            assert gap < error / 100, (n, stage + 1, rec, gap, error)
            dims[rec].append(k)
            if rec:
                _, W = linalg.eigh(V.T @ (A @ V))
                U = V @ W[:, :4]
        print(f"n={n}, stage={stage + 1}: {dims[0][-1]}, {dims[1][-1]}", flush=True)
    return dims


def resolvent(diag, b, order):
    z, q = special.roots_legendre(order)
    theta = (z + 1) * np.pi / 4
    out = np.zeros(len(b))
    ab = np.zeros((3, len(b)))
    ab[0, 1:] = -1
    ab[2, :-1] = -1
    for th, weight in zip(theta, q, strict=True):
        ab[1, :] = diag + np.tan(th) ** 2
        out += (weight / 2 / np.cos(th) ** 2) * linalg.solve_banded((1, 1), ab, b)
    return out


def scan(A, b, K, U, S, O, rule, ell, u, x_ref):
    t, w = rule["t"], rule["w"]
    accepted = {}
    an = frobenius_upper(A.data)
    for k in KS:
        V = basis(np.column_stack((K[:, :k], U)))
        AV = A @ V
        Q, R = np.linalg.qr(S @ V, mode="reduced")
        AZ = linalg.solve_triangular(R.T, AV.T, lower=True).T
        H, g = Q.T @ S @ AZ, Q.T @ S @ b
        I = np.eye(len(g))
        ys = np.column_stack([linalg.solve(H + tj * I, g) for tj in t])
        coords = linalg.solve_triangular(R, ys)
        summed = coords @ w
        x = V @ summed
        vn = frobenius_upper(V)
        beta_av = multiply_up(gamma_p(3, U64), multiply_up(an, vn))
        norms = [frobenius_upper(c) for c in coords.T]
        rho = [multiply_up(beta_av, nc) for nc in norms]
        eout = add_up(
            prod_radius(V, summed, ops=1),
            multiply_up(vn, prod_radius(coords, w, ops=1)),
        )

        for method, sketch in zip(METHODS, (None, O), strict=True):
            if method in accepted:
                continue
            rb = residual_bounds(b, AV, V, coords, t, norms, sketch)
            acc = iv.mpf(0)
            for tj, wj, norm, radius in zip(t, w, rb, rho, strict=True):
                tj_iv = ivfloat(tj)
                lam = ivfloat(float(np.clip(tj, ell, u)))
                acc += (
                    ivfloat(wj)
                    * iv.sqrt(lam)
                    / (lam + tj_iv)
                    * ivfloat(add_up(norm, radius))
                )
            eta = interval_upper(
                ivfloat(rule["eta_sinc"]) + acc + iv.sqrt(ivfloat(u)) * ivfloat(eout)
            )
            if eta <= TAU:
                error = normA(A, x_ref - x)
                assert error <= eta * (1 + 1e-8)
                accepted[method] = (k, error)
                if method == "direct":
                    V_direct = V
        if len(accepted) == len(METHODS):
            break

    assert accepted.keys() == set(METHODS)
    k, err_direct = accepted["direct"]
    k_sketch, err_sketch = accepted["finite-vector"]
    assert k == k_sketch
    return k_sketch, V_direct, min(err_direct, err_sketch)


def residual_bounds(b, AV, V, coords, t, norms, O):
    eb = ea = ev = 0.0
    if O is not None:
        eb = prod_radius(O, b, ops=1)
        ea = prod_radius(O, AV, ops=1)
        ev = prod_radius(O, V, ops=1)
        b, AV, V = O @ b, O @ AV, O @ V
    first = AV @ coords
    scaled = (V @ coords) * t
    left = b[:, None] - first
    res = left - scaled
    gamma = gamma_p(coords.shape[0], U64)
    ga = multiply_up(gamma, frobenius_upper(AV))
    gv = multiply_up(gamma, frobenius_upper(V))
    gs = _divide_up(U64, _next_down_positive(1 - U64))
    den = float(np.nextafter(np.sqrt(1 - EPS), -np.inf))
    result = []
    for j, (tj, nc) in enumerate(zip(t, norms, strict=True)):
        radius = add_up(
            multiply_up(ga, nc),
            multiply_up(tj, multiply_up(gv, nc)),
            multiply_up(gs, frobenius_upper(scaled[:, j])),
            sub_radius(b, first[:, j], ops=1),
            sub_radius(left[:, j], scaled[:, j], ops=1),
        )
        if O is not None:
            radius = add_up(
                radius, eb, multiply_up(add_up(ea, multiply_up(tj, ev)), nc)
            )
        norm = add_up(frobenius_upper(res[:, j]), radius)
        result.append(_divide_up(norm, den) if O is not None else norm)
    return result


if __name__ == "__main__":
    main()
