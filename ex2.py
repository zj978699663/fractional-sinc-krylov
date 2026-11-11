import numpy as np
from threadpoolctl import threadpool_limits

from common import (
    BUILD,
    norm_up,
    normA,
    reference,
    rounded,
    sinc,
    spectral_enclosure,
    write_table,
)


def sci(value, upward=False):
    number = rounded(value, upward=upward)
    exponent = number.adjusted()
    return f"{number.scaleb(-exponent):.2f}e{exponent:+d}"


def sinc_sweep():
    rng = np.random.default_rng(20261010)
    Q, _ = np.linalg.qr(rng.normal(size=(48, 48)))
    b = Q @ np.linspace(0.5, 1.5, 48)
    rows = []
    for ratio in [10.0, 100.0]:
        eig = np.geomspace(1.0, ratio, 48)
        A = (Q * eig) @ Q.T
        A = (A + A.T) / 2
        ell, u = spectral_enclosure(A)
        for alpha in [0.25, 0.5, 0.75]:
            xref, gap = reference(A, Q, eig, b, alpha)
            for tau in [1e-3, 1e-5, 1e-7]:
                r = sinc(alpha, ell, u, norm_up(b), tau)
                xs = Q @ ((Q.T @ b) * np.sum(r["w"] / (eig[:, None] + r["t"]), axis=1))
                err = normA(A, xref - xs)
                assert gap < err * 0.01 and err <= r["eta_sinc"]
                rows.append(
                    dict(
                        alpha=alpha,
                        ell=1.0,
                        u=ratio,
                        tau=tau,
                        h=r["h"],
                        M=r["M"],
                        N=r["N"],
                        nodes=r["nodes"],
                        eta_sinc=r["eta_sinc"],
                        error_sinc=err,
                        ref_gap=gap,
                    )
                )
    return rows


def main():
    BUILD.mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        sweep = sinc_sweep()
    cases = {(r["u"], r["alpha"], r["tau"]): r for r in sweep}
    assert len(cases) == 18
    rows = []
    for alpha in (0.25, 0.5, 0.75):
        for tau in (1e-3, 1e-5, 1e-7):
            row = [alpha, tau]
            for u in (10, 100):
                r = cases[u, alpha, tau]
                assert r["error_sinc"] <= r["eta_sinc"] <= 0.2 * tau
                assert r["nodes"] == r["M"] + r["N"] + 1
                row.extend((r["nodes"], sci(r["error_sinc"]), sci(r["eta_sinc"], True)))
            rows.append(row)
    write_table(
        2,
        [
            "alpha",
            "tau",
            "nodes_u10",
            "error_u10",
            "eta_u10",
            "nodes_u100",
            "error_u100",
            "eta_u100",
        ],
        rows,
    )


if __name__ == "__main__":
    main()
