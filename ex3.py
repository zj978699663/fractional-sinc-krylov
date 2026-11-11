from decimal import Decimal

import numpy as np
from scipy import linalg as la
from threadpoolctl import threadpool_limits

from common import (
    BUILD,
    DELTA,
    EPS,
    axes,
    basis,
    fixed,
    finite_matrix,
    matrix_roundoff,
    rank_two_output_error,
    interval_upper,
    ivfloat,
    normA,
    plotting,
    projection,
    recycle,
    rounded,
    save,
    sinc,
    write_table,
)


def scaled(value, upward=False):
    number = rounded(Decimal(str(value)) * 100, upward=upward)
    return f"{number:.{max(0, 2 - number.adjusted())}f}"


def combine_certificates(eta_res, eta_matrix):
    return min(
        eta_res,
        eta_matrix if np.isfinite(eta_matrix) and eta_matrix >= 0 else float("inf"),
    )


def matrix_reference_checks(rows):
    for row in rows:
        mu = float(row["mu"])
        alpha = float(row["alpha"])
        H = np.array([[2.0, 1.0], [1.0, 2.0]])
        A = np.block([[H, mu * np.eye(2)], [mu * np.eye(2), H]])
        Q = (
            np.array(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, -1.0, -1.0],
                    [1.0, -1.0, 1.0, -1.0],
                    [1.0, -1.0, -1.0, 1.0],
                ]
            )
            / 2
        )
        eig = np.array([3 + mu, 3 - mu, 1 + mu, 1 - mu])
        b = np.eye(4)[:, 0]
        rule = sinc(alpha, 1 - mu, 3 + mu, 1.0, 1e-5)
        q = lambda lam: np.sum(rule["w"] / (lam[:, None] + rule["t"]), axis=1)
        exact = Q @ ((Q.T @ b) * q(eig))
        ev, X = la.eigh(A)
        reference = X @ ((X.T @ b) * q(ev))
        gap = normA(A, exact - reference)
        assert gap < 0.01 * float(row["error_alg"])
        row["reference_gap"] = gap
    return rows


def matrix_cases():
    rows = []
    rng = np.random.default_rng(20261009)
    for mu in [0.125, 0.25, 0.5]:
        H = np.array([[2.0, 1.0], [1.0, 2.0]])
        A = np.block([[H, mu * np.eye(2)], [mu * np.eye(2), H]])
        b = np.eye(4)[:, 0]
        V = np.eye(4)[:, :2]
        S = np.eye(4)[:2]
        ell, u = 1 - mu, 3 + mu
        nu = int(
            np.ceil(
                ((2 + np.sqrt(2 * np.log(2 / DELTA))) / (np.sqrt(1 + EPS) - 1)) ** 2
            )
        )
        Omega = rng.normal(size=(nu, 4)) / np.sqrt(nu)
        G = Omega[:, 2:].T @ Omega[:, 2:]
        assert (
            la.eigvalsh(Omega.T @ Omega).min() >= 1 - EPS
            and la.eigvalsh(Omega.T @ Omega).max() <= 1 + EPS
        )
        for alpha in [0.25, 0.5, 0.75]:
            prior_rng = np.random.default_rng(20261011 + int(1000 * mu + 100 * alpha))
            Ap = np.eye(4)
            bp = np.eye(4)[:, 1]
            Vp = bp[:, None]
            Sp = prior_rng.normal(size=(2, 4)) / np.sqrt(2)
            Op = prior_rng.normal(size=(nu, 4)) / np.sqrt(nu)
            prior_rule = sinc(alpha, 1.0, 1.0, 1.0, 1e-5)
            prior = projection(
                Ap, bp, Vp, Sp, prior_rule["t"], prior_rule["w"], 1.0, 1.0, Op
            )
            prior_eta = prior_rule["eta_sinc"] + prior["eta_res"]
            assert normA(Ap, bp - prior["x"]) <= prior_eta <= 1e-5
            Up, prior_schur = recycle(Ap, Vp, Sp, 1)
            pivot = int(np.argmax(abs(Up[:, 0])))
            Up = Up / Up[pivot, 0]
            V = basis(np.column_stack((b, Up[:, 0].real)))
            assert np.array_equal(V, np.eye(4)[:, :2])
            rule = sinc(alpha, ell, u, 1.0, 1e-5)
            pr = projection(A, b, V, S, rule["t"], rule["w"], ell, u, Omega)
            c = -mu * pr["ys"]
            theta, coefferr = matrix_roundoff(
                Omega, G, mu, c, rule["t"], rule["w"], ell, u
            )
            ev, X = la.eigh(A)
            xs = X @ ((X.T @ b) * np.sum(rule["w"] / (ev[:, None] + rule["t"]), axis=1))
            ea = normA(A, xs - pr["x"])
            output_error = rank_two_output_error(rule["t"], rule["w"], pr["x"], u)
            for cells in (
                [32, 128, 512, 2048] if mu == 0.25 and alpha == 0.5 else [512]
            ):
                result = finite_matrix(
                    c, rule["t"], rule["w"], ell, u, G, cells, theta, coefferr
                )
                result["eta_matrix"] = interval_upper(
                    ivfloat(result["eta_matrix"]) + ivfloat(output_error)
                )
                result["ratio"] = result["eta_matrix"] / result["eta_triangle"]
                result["projected_output_error"] = output_error
                row = dict(
                    experiment="matrix",
                    mu=mu,
                    alpha=alpha,
                    n=4,
                    m=1,
                    k=1,
                    ell=ell,
                    u=u,
                    nu=nu,
                    nodes=rule["nodes"],
                    error_alg=ea,
                    eta_res=pr["eta_res"],
                    **result,
                )
                row.update(
                    prior_eta_total=prior_eta,
                    prior_error=normA(Ap, bp - prior["x"]),
                    prior_schur_gap=prior_schur,
                    defect_rank=int(np.linalg.matrix_rank(pr["F"])),
                )
                row["eta_refined"] = combine_certificates(
                    pr["eta_res"], result["eta_matrix"]
                )
                row["eta_sinc"] = rule["eta_sinc"]
                row["witness_tau"] = (
                    rule["eta_sinc"] + (result["eta_matrix"] + pr["eta_res"]) / 2
                    if result["eta_matrix"] < pr["eta_res"]
                    else ""
                )
                assert (
                    ea <= result["eta_matrix"]
                    and abs(pr["eta_res"] - result["eta_triangle"]) < 1e-12
                )
                rows.append(row)
                print(
                    "matrix", mu, alpha, cells, "ratio", result["ratio"], flush=True
                )

    A = np.diag([1.0, 2.0, 3.0, 4.0])
    b = np.ones(4)
    uvec = np.array([-1, 43 / 153, 1, 25 / 17])
    Z = np.column_stack((b, uvec))
    H = np.array([[2.0, 1.0], [1.0, 2.0]])
    t = np.array([1.0, 4.0])
    w = np.log(4) / np.pi * np.array([1.0, 2.0])
    qA = np.sum(w / (np.diag(A)[:, None] + t), axis=1) * b
    x = Z @ sum(
        (
            wj * la.solve(H + tj * np.eye(2), np.array([1.0, 0.0]))
            for tj, wj in zip(t, w)
        ),
        np.zeros(2),
    )
    ea = normA(A, qA - x)
    F = A @ Z - Z @ H
    c = np.column_stack(
        [-la.solve(H + tj * np.eye(2), np.array([1.0, 0.0])) for tj in t]
    )
    v = np.sqrt(2.0) * np.sum(c * w / (2.0 + t), axis=1)
    lower = np.sqrt(v @ (F.T @ F) @ v)
    assert ea < 1e-13 and lower > 0
    return matrix_reference_checks(rows)


def draw_matrix(data):
    plt = plotting()
    from matplotlib.ticker import FixedLocator, FuncFormatter

    grid = sorted(
        [r for r in data if r["mu"] == 0.25 and r["alpha"] == 0.5],
        key=lambda r: r["cells"],
    )
    fig, ax = axes(plt, "single")
    ax.set_xscale("log", base=2)
    ax.set_yscale("linear")
    x = [int(r["cells"]) for r in grid]
    y = [float(r["eta_matrix"]) / float(r["eta_res"]) for r in grid]
    ax.plot(x, y, color="black", marker="o", linestyle="-")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.7)
    ax.set(
        xlim=(24, 2800),
        ylim=(0.96, 1.012),
        xlabel="Cells",
        ylabel=r"$\eta^{\mathrm{mat}}/\eta^{\mathrm{res}}$",
    )
    ax.xaxis.set_major_locator(FixedLocator(x))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
    save(plt, fig, ax, "fig2.pdf", "single")


def main():
    BUILD.mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        data = matrix_cases()
    cases = [r for r in data if r["cells"] == 512]
    assert len(cases) == 9
    rows = []
    for name, key, upward in (
        ("error_alg", "error_alg", False),
        ("eta_res", "eta_res", True),
        ("eta_mat", "eta_matrix", True),
    ):
        rows.append([name, *(scaled(r[key], upward) for r in cases)])
    rows.append(
        [
            "ratio",
            *(
                fixed(Decimal(str(r["eta_matrix"])) / Decimal(str(r["eta_res"])), 4)
                for r in cases
            ),
        ]
    )
    assert all(r["eta_matrix"] < r["eta_res"] for r in cases)
    header = ["quantity", *(f"mu{r['mu']:g}_alpha{r['alpha']:g}" for r in cases)]
    write_table(3, header, rows)
    draw_matrix(data)


if __name__ == "__main__":
    main()
