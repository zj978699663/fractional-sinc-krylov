from decimal import Decimal
import numpy as np
from scipy import linalg as la
from threadpoolctl import threadpool_limits

from common import (
    BUILD,
    COLORS,
    DELTA,
    EPS,
    LINES,
    MARKERS,
    PRESETS,
    axes,
    basis,
    fixed,
    krylov,
    normA,
    norm_up,
    plotting,
    projection,
    recycle,
    reference,
    rounded,
    save,
    sinc,
    spectral_enclosure,
    write_table,
)

SEED = 20260816


def zero_rhs(b):
    return (
        dict(x=np.zeros_like(b), eta_total=0.0, status="zero_rhs")
        if not np.any(b)
        else None
    )


def first_accept(rows, tau):
    return next(
        (r for r in rows if r.get("status") == "ok" and r["eta_total"] <= tau), None
    )


def sequence(complex_input=False):
    rng = np.random.default_rng(SEED + (100 if complex_input else 0))
    n = 72
    p = 5 if not complex_input else 3
    U = np.empty((n, 0))
    paths = []
    finals = []
    for i in range(p):
        eig = np.geomspace(1.0, 10.0 + 5 * i, n)
        G = rng.normal(size=(n, n))
        if complex_input:
            G = G + 1j * rng.normal(size=(n, n))
        Q, _ = np.linalg.qr(G)
        A = (Q * eig) @ Q.conj().T
        A = (A + A.conj().T) / 2
        coeff = np.cos(np.linspace(0, 3, n) + 0.35 * i) + 0.2 * np.sin(
            np.linspace(0, 7, n)
        )
        if complex_input:
            coeff = coeff + 1j * 0.3 * np.sin(np.linspace(0, 5, n) + i)
        b = Q @ coeff
        alpha = 0.5
        tau = 2e-5
        ell, u = spectral_enclosure(A)
        if zero_rhs(b) is not None:
            raise RuntimeError("unexpected zero test input")
        rule = sinc(alpha, ell, u, norm_up(b), tau)
        S = rng.normal(size=(56, n)) / np.sqrt(56)
        candidates = [4, 8, 12, 16, 20, 24, 28, 32]
        bases = {m: basis(np.column_stack((krylov(A, b, m), U))) for m in candidates}
        dmax = max(v.shape[1] for v in bases.values())
        nu = int(
            np.ceil(
                (
                    (np.sqrt(min(n, 2 + 4 * dmax)) + np.sqrt(2 * np.log(2 / DELTA)))
                    / (np.sqrt(1 + EPS) - 1)
                )
                ** 2
            )
        )

        Omega = rng.normal(size=(nu, n)) / np.sqrt(nu)
        xref, refgap = reference(A, Q, eig, b, alpha)
        qval = np.sum(rule["w"][None, :] / (eig[:, None] + rule["t"][None, :]), axis=1)
        xsinc = Q @ ((Q.conj().T @ b) * qval)
        W = np.column_stack((b, bases[32], A @ bases[32]))
        Wreal = basis(np.column_stack((W.real, W.imag)))
        eigenval = la.eigvalsh((Omega @ Wreal).T @ (Omega @ Wreal))
        embedding = bool(eigenval.min() >= 1 - EPS and eigenval.max() <= 1 + EPS)
        first = None
        for m in candidates:
            common = dict(
                experiment="complex" if complex_input else "sequence",
                problem=i + 1,
                n=n,
                m=m,
                k=U.shape[1],
                s=56,
                nu=nu,
                alpha=alpha,
                ell=ell,
                u=u,
                tau=tau,
                seed=SEED + (100 if complex_input else 0),
                h=rule["h"],
                M=rule["M"],
                N=rule["N"],
                nodes=rule["nodes"],
                epsilon=EPS,
                delta=DELTA,
                ref_gap=refgap,
            )
            try:
                obj = projection(A, b, bases[m], S, rule["t"], rule["w"], ell, u, Omega)
            except la.LinAlgError as err:
                paths.append(dict(common, status=str(err), accepted=0))
                continue
            err = normA(A, xref - obj["x"])
            eta = rule["eta_sinc"] + obj["eta_res"]
            es = normA(A, xref - xsinc)
            ea = normA(A, xsinc - obj["x"])
            row = dict(
                common,
                status="ok",
                eta_sinc=rule["eta_sinc"],
                eta_res=obj["eta_res"],
                eta_total=eta,
                error_total=err,
                error_sinc=es,
                error_alg=ea,
                effectivity=eta / err,
                accepted=int(eta <= tau),
                embedding=embedding,
                flow_gap=obj["flow_gap"],
                defect_gap=obj["defect_gap"],
                qr_condition=obj["qr_condition"],
            )
            assert err <= eta * (1 + 1e-10) and es <= rule["eta_sinc"] * (1 + 1e-8)
            assert refgap < 0.01 * err
            paths.append(row)
            if first is None and first_accept([row], tau) is not None:
                first = (m, obj, row)
        if first is None:
            finals.append(dict(common, status="no_return"))
            continue
        m, obj, row = first
        finals.append(row)
        V = bases[m]
        for _ in range(5):
            rng.normal(size=(nu, n))
        U, _ = recycle(A, V, S, 6)
        print(common["experiment"], i + 1, "m", m, "eta", row["eta_total"], flush=True)
    return paths, finals


def draw_sequences(paths):
    plt = plotting()
    paths = [r for r in paths if r["experiment"] == "sequence"]
    low = min(float(r["error_total"]) for r in paths) / 2
    high = max(float(r["eta_total"]) for r in paths) * 2.0
    tolerance = 2e-5
    for name, field, ylabel in [
        ("fig1a.pdf", "error_total", r"$E$"),
        ("fig1b.pdf", "eta_total", r"$\eta^{\mathrm{tot}}$"),
    ]:
        fig, ax = axes(plt, "pair")
        ax.set_yscale("log")
        accepted = {}
        for i in range(5):
            z = sorted(
                [r for r in paths if int(r["problem"]) == i + 1],
                key=lambda r: int(r["m"]),
            )
            x, y = [int(r["m"]) for r in z], [float(r[field]) for r in z]
            assert all(3 < xx < 33 and low < yy < high for xx, yy in zip(x, y))
            ax.plot(
                x,
                y,
                color=COLORS[i],
                marker=MARKERS[i],
                linestyle=LINES[i],
                label=rf"$i={i + 1}$",
                zorder=3 + i,
            )
            selected = next(r for r in z if float(r["eta_total"]) <= tolerance)
            accepted[str(i + 1)] = int(selected["m"])
            if field == "eta_total":
                ax.plot(
                    int(selected["m"]),
                    float(selected["eta_total"]),
                    ls="none",
                    marker=MARKERS[i],
                    color=COLORS[i],
                    ms=5.3,
                    mew=1.0,
                    zorder=10 + i,
                )
        ax.axhline(tolerance, color="black", ls=":", lw=0.7, zorder=2)
        ax.set(
            xlim=(3, 33),
            ylim=(low, high),
            xticks=[4, 12, 20, 28, 32],
            xlabel=r"$m$",
            ylabel=ylabel,
        )
        legend = ax.legend(
            loc="upper right",
            ncol=2,
            frameon=True,
            facecolor="white",
            framealpha=0.8,
            edgecolor=".4",
            handlelength=1.2,
            columnspacing=0.65,
            labelspacing=0.22,
            borderpad=0.28,
            handletextpad=0.45,
        )
        legend.get_frame().set_linewidth(PRESETS["pair"]["legend_edge"])
        ax.text(4.2, tolerance * 1.5, r"$\tau$", fontsize=7.2)
        save(plt, fig, ax, name, "pair")


def main():
    BUILD.mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        real, real_final = sequence()
        complex_paths, complex_final = sequence(True)
    paths = real + complex_paths
    accepted = real_final + complex_final
    assert len(paths) == 64 and len(accepted) == 8
    rows = []
    for r in paths:
        assert r["status"] == "ok"
        assert r["error_total"] <= r["eta_total"]
        assert r["error_sinc"] <= r["eta_sinc"] <= 0.2 * r["tau"]
    for r in accepted:
        label = "Real" if r["experiment"] == "sequence" else "Complex"
        rows.append(
            [
                label,
                r["problem"],
                r["m"],
                str(rounded(Decimal(str(r["eta_sinc"])) * 1000000, upward=True)),
                str(rounded(Decimal(str(r["eta_total"])) * 1000000, upward=True)),
                str(rounded(Decimal(str(r["error_total"])) * 1000000)),
                fixed(r["effectivity"], 2),
            ]
        )
    write_table(
        1,
        [
            "input",
            "stage",
            "m",
            "eta_sinc_1e-6",
            "eta_total_1e-6",
            "error_1e-6",
            "effectivity",
        ],
        rows,
    )
    draw_sequences(paths)


if __name__ == "__main__":
    main()
