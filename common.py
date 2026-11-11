import csv
import math
import os
import tempfile
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_HALF_EVEN
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy import linalg as la

linalg = la


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
REFERENCE = ROOT / "reference"
DATA = ROOT / "data"
EPS, DELTA, GAMMA = 0.2, 0.01, 0.2
U64 = 2.0**-53
iv = mp.iv
iv.dps = 35


def value(cell):
    try:
        return Decimal(cell)
    except InvalidOperation:
        return cell


def write_table(n, header, rows):
    name = f"table{n}.csv"
    actual = [[str(cell) for cell in row] for row in [header, *rows]]
    with (REFERENCE / name).open(encoding="utf-8", newline="") as stream:
        expected = list(csv.reader(stream))
    if len(actual) != len(expected):
        raise ValueError(
            f"{name}: expected {len(expected) - 1} rows, got {len(actual) - 1}"
        )
    for i, (a, b) in enumerate(zip(actual, expected), 1):
        if list(map(value, a)) != list(map(value, b)):
            raise ValueError(f"{name}, line {i}: {a} differs from {b}")
    with (BUILD / name).open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(actual)
    print(f"{name}: {len(actual) - 1} rows checked")


def rounded(value, digits=3, upward=False):
    number = Decimal(str(value))
    quantum = Decimal(1).scaleb(number.adjusted() - digits + 1)
    mode = ROUND_CEILING if upward else ROUND_HALF_EVEN
    return number.quantize(quantum, rounding=mode)


def fixed(value, places=6, upward=False):
    mode = ROUND_CEILING if upward else ROUND_HALF_EVEN
    return f"{Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=mode):.{places}f}"


def _next_up(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise FloatingPointError(f"invalid nonnegative bound value: {value}")
    return float(np.nextafter(value, np.inf))


def _next_down_positive(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or not value > 0.0:
        raise FloatingPointError(f"invalid positive bound value: {value}")
    lower = float(np.nextafter(value, -np.inf))
    if not lower > 0.0:
        raise FloatingPointError("positive lower enclosure underflowed")
    return lower


def _add_up(first: float, second: float) -> float:
    first = float(first)
    second = float(second)
    if not np.isfinite(first) or not np.isfinite(second) or first < 0.0 or second < 0.0:
        raise FloatingPointError("invalid nonnegative addition operands")
    value = first + second
    if value == 0.0 and (first != 0.0 or second != 0.0):
        raise FloatingPointError("underflow in nonnegative addition")
    if value == 0.0:
        return 0.0
    return _next_up(value)


def multiply_up(first: float, second: float) -> float:
    first = float(first)
    second = float(second)
    if not np.isfinite(first) or not np.isfinite(second) or first < 0.0 or second < 0.0:
        raise FloatingPointError("invalid nonnegative product operands")
    value = first * second
    if value == 0.0 and first != 0.0 and second != 0.0:
        raise FloatingPointError("underflow in nonnegative product")
    if value == 0.0:
        return 0.0
    return _next_up(value)


def _divide_up(numerator: float, denominator: float) -> float:
    numerator = float(numerator)
    denominator = float(denominator)
    if (
        not np.isfinite(numerator)
        or numerator < 0.0
        or not np.isfinite(denominator)
        or not denominator > 0.0
    ):
        raise FloatingPointError(f"invalid positive denominator: {denominator}")
    value = numerator / denominator
    if value == 0.0 and numerator != 0.0:
        raise FloatingPointError("underflow in nonnegative division")
    if value == 0.0:
        return 0.0
    return _next_up(value)


def add_up(*values: float) -> float:
    total = 0.0
    for value in values:
        if not np.isfinite(value) or value < 0.0:
            raise FloatingPointError(f"invalid nonnegative summand: {value}")
        total = _add_up(total, float(value))
    return total


def sqrt_up(value: float) -> float:
    if not np.isfinite(value) or value < 0.0:
        raise FloatingPointError(f"invalid square-root upper operand: {value}")
    if value == 0.0:
        return 0.0
    return _next_up(math.sqrt(float(value)))


def gamma_p(p: int, unit_roundoff: float) -> float:
    value = multiply_up(float(p), unit_roundoff)
    if not value < 1.0:
        raise FloatingPointError(f"invalid rounding bound: p*u={value}")
    return _divide_up(value, _next_down_positive(1.0 - value))


def frobenius_upper(array: np.ndarray) -> float:
    values = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("non-finite operand in Frobenius bound")
    count = int(values.size)
    if count == 0:
        return 0.0

    square_safe_minimum = math.sqrt(np.finfo(np.float64).tiny)
    if np.any((values != 0.0) & (np.abs(values) < square_safe_minimum)):
        raise FloatingPointError("underflow risk in Frobenius bound")
    squares = values * values
    stored_sum = float(np.sum(squares, dtype=np.float64))
    if stored_sum == 0.0 and np.any(values != 0.0):
        raise FloatingPointError("underflow in Frobenius bound")
    if not np.isfinite(stored_sum):
        raise FloatingPointError("overflow in Frobenius bound")
    gamma_sum = gamma_p(2 * count, U64)
    sumsq_up = _divide_up(
        stored_sum, _next_down_positive(1.0 - gamma_sum)
    )
    return sqrt_up(sumsq_up)


def normA(A, e):
    return float(np.sqrt(max(0.0, np.vdot(e, A @ e).real)))


def basis(B):
    cols = []
    for col in B.T:
        v = col.astype(np.result_type(B, float), copy=True)
        for _ in range(2):
            for q in cols:
                v -= q * np.vdot(q, v)
        nv = np.linalg.norm(v)
        if nv > 1e-12 * max(1.0, np.linalg.norm(col)):
            cols.append(v / nv)
    if not cols:
        raise la.LinAlgError("empty search space")
    return np.column_stack(cols)


def krylov(A, b, m):
    q = b / np.linalg.norm(b)
    cols = []
    for _ in range(m):
        v = q.copy()
        for z in cols:
            v -= z * np.vdot(z, v)
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            break
        q = v / nv
        cols.append(q)
        q = A @ q
    return np.column_stack(cols)


def recycle(A, V, S, k):
    Q, R = np.linalg.qr(S @ V, mode="reduced")
    H = Q.conj().T @ S @ A @ la.solve_triangular(R.T, V.T, lower=True).T
    Mhat = la.solve_triangular(R, H @ R)
    T, X = la.schur(Mhat, output="complex")
    trexc = la.get_lapack_funcs("trexc", (T,))
    kk = min(k, len(T))
    for j in range(kk):
        jmin = min(
            range(j, len(T)),
            key=lambda z: (abs(T[z, z]), T[z, z].real, T[z, z].imag, z),
        )
        if jmin != j:
            T, X, info = trexc(T, X, jmin + 1, j + 1)
            if info:
                raise RuntimeError("Schur reordering failed")
    gap = float(np.linalg.norm(Mhat @ X[:, :kk] - X[:, :kk] @ T[:kk, :kk]))
    U = np.linalg.qr(V @ X[:, :kk], mode="reduced")[0]
    return U, gap


def reference(A, Q, eig, b, alpha):
    x = Q @ ((Q.conj().T @ b) * eig ** (-alpha))
    vals, vec = la.eigh(A)
    y = vec @ ((vec.conj().T @ b) * vals ** (-alpha))
    return x, normA(A, x - y)


def interval_upper(x):
    return np.nextafter(float(x.b), np.inf)


def interval_lower(x):
    return np.nextafter(float(x.a), -np.inf)


def ivfloat(x):
    return iv.mpf(float(x))


def sinc_rule(alpha, ell, u, bnorm, tau):
    c = np.sin(np.pi * alpha) / np.pi
    theta = GAMMA * tau / bnorm
    chi = max(ell ** (0.5 - alpha), u ** (0.5 - alpha))
    rho = min(0.5, theta / (32 * np.sin(np.pi * alpha) * chi))
    h = 2 * np.pi**2 / np.log(1 / rho)
    M = max(
        0,
        int(
            np.ceil(
                np.log(
                    3 * c * h * ell ** (-0.5) / (theta * (-np.expm1(-(1 - alpha) * h)))
                )
                / ((1 - alpha) * h)
                - 1
            )
        ),
    )
    N = max(
        0,
        int(
            np.ceil(
                np.log(3 * c * h * u**0.5 / (theta * (-np.expm1(-alpha * h))))
                / (alpha * h)
                - 1
            )
        ),
    )
    jj = np.arange(-M, N + 1)
    t = np.exp(jj * h)
    w = c * h * np.exp((1 - alpha) * jj * h)
    D = 4 * np.sin(np.pi * alpha) * rho / ((1 - rho) * (1 - rho * rho))
    eta = bnorm * (
        D * chi
        + c
        * h
        * ell ** (-0.5)
        * np.exp(-(1 - alpha) * (M + 1) * h)
        / (-np.expm1(-(1 - alpha) * h))
        + c * h * u**0.5 * np.exp(-alpha * (N + 1) * h) / (-np.expm1(-alpha * h))
    )
    assert np.isfinite(t).all() and np.isfinite(w).all() and np.min(w) > 0
    assert eta <= GAMMA * tau * (1 + 1e-12)
    return dict(h=h, M=M, N=N, t=t, w=w, eta_sinc=eta, nodes=len(t), theta=theta)


def norm_up(x):
    z = np.asarray(x)
    return frobenius_upper(np.r_[z.real.ravel(), z.imag.ravel()])


def prod_radius(a, b, ops=8):
    norm = frobenius_upper if ops == 1 else norm_up
    return multiply_up(
        gamma_p(ops * a.shape[1], U64), multiply_up(norm(a), norm(b))
    )


def weighted_sum_radius(a, b):
    g = gamma_p(8 * a.shape[1], U64)
    positive = np.abs(a) @ np.abs(b)
    magnitude = _divide_up(
        norm_up(positive), _next_down_positive(1 - gamma_p(16 * a.shape[1], U64))
    )
    return multiply_up(g, magnitude)


def sub_radius(a, b, ops=8):
    norm = frobenius_upper if ops == 1 else norm_up
    return multiply_up(ops * U64, add_up(norm(a), norm(b)))


def spectral_enclosure(A):
    values, V = linalg.eigh(A)
    identity = np.eye(len(A))
    gram = V.conj().T @ V
    defect = gram - identity
    theta = add_up(
        norm_up(defect), prod_radius(V.conj().T, V), sub_radius(gram, identity)
    )
    assert theta < 1
    AV = A @ V
    VD = V * values
    residual = AV - VD
    rad = add_up(
        norm_up(residual),
        prod_radius(A, V),
        prod_radius(V, np.diag(values)),
        sub_radius(AV, VD),
    )
    smin_lo = float(
        np.nextafter(np.sqrt(float(np.nextafter(1 - theta, -np.inf))), -np.inf)
    )
    error = _divide_up(rad, smin_lo)
    ell = float(np.nextafter(values[0] - error, -np.inf))
    u = float(np.nextafter(values[-1] + error, np.inf))
    assert ell > 0
    return ell, u


def sinc(alpha, ell, u, bnorm, tau):
    r = sinc_rule(alpha, ell, u, bnorm, tau)
    r["eta_sinc"] = sinc_bound(r, alpha, ell, u, bnorm)
    return r


def sinc_bound(r, alpha, ell, u, bnorm):
    aa = ivfloat(alpha)
    hh = ivfloat(r["h"])
    ll = ivfloat(ell)
    uu = ivfloat(u)
    cc = iv.sin(iv.pi * aa) / iv.pi
    rho = iv.exp(-2 * iv.pi**2 / hh)
    chi = max(interval_upper(ll ** (0.5 - aa)), interval_upper(uu ** (0.5 - aa)))
    value = 4 * iv.sin(iv.pi * aa) * rho / ((1 - rho) * (1 - rho * rho)) * ivfloat(chi)
    value += (
        cc
        * hh
        / iv.sqrt(ll)
        * iv.exp(-(1 - aa) * (r["M"] + 1) * hh)
        / (1 - iv.exp(-(1 - aa) * hh))
    )
    value += (
        cc * hh * iv.sqrt(uu) * iv.exp(-aa * (r["N"] + 1) * hh) / (1 - iv.exp(-aa * hh))
    )
    for j, t, w in zip(range(-r["M"], r["N"] + 1), r["t"], r["w"]):
        tj = iv.exp(j * hh)
        wj = cc * hh * iv.exp((1 - aa) * j * hh)
        value += abs(ivfloat(w) - wj) * iv.sqrt(uu) / (ll + tj)
        value += (
            ivfloat(w)
            * abs(ivfloat(t) - tj)
            * iv.sqrt(uu)
            / ((ll + tj) * (ll + ivfloat(t)))
        )
    return interval_upper(value * ivfloat(bnorm))


def projection(A, b, V, S, t, w, ell, u, Omega, epsilon=EPS):
    AV = A @ V
    SV = S @ V
    if SV.shape[0] < SV.shape[1] or np.linalg.matrix_rank(SV) < V.shape[1]:
        raise la.LinAlgError("rank_failure")
    Q, R = np.linalg.qr(SV, mode="reduced")
    Z = la.solve_triangular(R.T, V.T, lower=True).T
    AZ = la.solve_triangular(R.T, AV.T, lower=True).T
    H = Q.conj().T @ S @ AZ
    g = Q.conj().T @ S @ b
    Omegab = Omega @ b
    OmegaV = Omega @ V
    OmegaAV = Omega @ AV
    ys = []
    res = []
    sk = []
    for tj in t:
        shifted = H + tj * np.eye(len(g))
        if np.linalg.matrix_rank(shifted) < len(g):
            raise la.LinAlgError("shift_failure")
        y = la.solve(shifted, g)
        zcoord = la.solve_triangular(R, y)

        rr = b - AV @ zcoord - tj * (V @ zcoord)
        ss = Omegab - (OmegaAV + tj * OmegaV) @ zcoord
        ys.append(y)
        res.append(rr)
        sk.append(ss)
    ys = np.column_stack(ys)
    res = np.column_stack(res)
    sk = np.column_stack(sk)
    x = Z @ (ys @ w)
    F = AZ - Z @ H
    obj = dict(
        x=x,
        H=H,
        g=g,
        Z=Z,
        F=F,
        ys=ys,
        res=res,
        sk=sk,
        flow_gap=float(np.linalg.norm(sk - Omega @ res)),
        defect_gap=float(np.linalg.norm(res + F @ ys)),
        qr_condition=float(np.linalg.cond(R)),
    )

    coords = linalg.solve_triangular(R, ys)
    oa, ov, ob = OmegaAV, OmegaV, Omegab
    ea = prod_radius(Omega, AV)
    ev = prod_radius(Omega, V)
    eb = prod_radius(Omega, b)
    eav = prod_radius(A, V)
    ga = gamma_p(8 * oa.shape[1], U64)
    na = norm_up(oa)
    nv = norm_up(ov)
    nb = norm_up(ob)
    total = iv.mpf(0)
    max_radius = 0.0
    for j, tj in enumerate(t):
        c = coords[:, j]
        first = oa @ c
        second = ov @ c
        scaled = tj * second
        left = ob - first
        res = left - scaled
        nc = norm_up(c)
        nf = norm_up(first)
        nl = norm_up(left)
        nscaled = norm_up(scaled)
        rad = add_up(
            eb,
            multiply_up(ga, multiply_up(na, nc)),
            multiply_up(ea, nc),
            multiply_up(
                tj, add_up(multiply_up(ga, multiply_up(nv, nc)), multiply_up(ev, nc))
            ),
            multiply_up(8 * U64, nscaled),
            multiply_up(8 * U64, add_up(nb, nf)),
            multiply_up(8 * U64, add_up(nl, nscaled)),
        )
        direct = multiply_up(eav, nc)
        ns = _divide_up(
            add_up(norm_up(res), rad),
            float(np.nextafter(np.sqrt(1 - epsilon), -np.inf)),
        )
        bound = add_up(ns, direct)
        max_radius = max(max_radius, rad)
        lam = ivfloat(float(np.clip(tj, ell, u)))
        tt = ivfloat(tj)
        total += ivfloat(w[j]) * iv.sqrt(lam) / (lam + tt) * ivfloat(bound)
    summed = coords @ w
    x2 = V @ summed
    output = add_up(
        norm_up(obj["x"] - x2),
        sub_radius(obj["x"], x2),
        prod_radius(V, summed),
        multiply_up(norm_up(V), weighted_sum_radius(coords, w)),
    )
    total += iv.sqrt(ivfloat(u)) * ivfloat(output)
    obj["eta_res"] = interval_upper(total)
    obj["stored_output_radius"] = output
    obj["maximum_sampled_residual_radius"] = max_radius
    return obj


TEXT_WIDTH_PT = 390.0
TEX_POINTS_PER_INCH = 72.27
COLORS = ("#176b91", "#ce6a22", "#16856b", "#845797", "#a63d52")
MARKERS = ("o", "s", "^", "D", "v")
LINES = ("-", "--", "-.", (0, (4, 1, 1, 1)), "-")
PRESETS = {
    "pair": dict(
        label=8.2,
        tick=7.2,
        legend=6.8,
        frame=0.55,
        line=0.80,
        marker=3.4,
        marker_edge=0.75,
        legend_edge=0.42,
        fraction=0.48,
    ),
    "single": dict(
        label=9.5,
        tick=8.5,
        legend=8.0,
        frame=0.60,
        line=0.90,
        marker=4.0,
        marker_edge=0.80,
        legend_edge=0.45,
        fraction=0.618,
    ),
}


def plotting():
    cache = BUILD / "cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    tempfile.tempdir = str(cache)
    import matplotlib as mpl

    mpl.use("pgf")
    import matplotlib.pyplot as plt

    return plt


def configure(plt, preset):
    p = PRESETS[preset]
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}",
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.unicode_minus": False,
            "font.size": p["label"],
            "axes.labelsize": p["label"],
            "xtick.labelsize": p["tick"],
            "ytick.labelsize": p["tick"],
            "legend.fontsize": p["legend"],
            "axes.linewidth": p["frame"],
            "lines.linewidth": p["line"],
            "lines.markersize": p["marker"],
            "lines.markeredgewidth": p["marker_edge"],
            "lines.markerfacecolor": "white",
            "axes.grid": False,
            "savefig.bbox": None,
            "savefig.pad_inches": 0,
        }
    )


def axes(plt, preset):
    configure(plt, preset)
    width = PRESETS[preset]["fraction"] * TEXT_WIDTH_PT / TEX_POINTS_PER_INCH
    if preset == "pair":
        height = width * 1.02
        rect = [0.18, 0.16, 0.80, 0.80 / 1.02]
        aspect = 1
    else:
        height = width * (0.83 / 1.5) / 0.825
        rect = [0.16, 0.16, 0.83, 0.825]
        aspect = 1 / 1.5
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes(rect)
    ax.set_box_aspect(aspect)
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        length=2.5,
        width=PRESETS[preset]["frame"],
        pad=2,
    )
    ax.tick_params(which="minor", length=1.5, width=0.4)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    return fig, ax


def save(plt, fig, ax, name, preset):
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    extent = fig.get_tightbbox(renderer)
    width, height = fig.get_size_inches()
    assert extent.x0 >= -0.001 and extent.y0 >= -0.001, (name, extent)
    assert extent.x1 <= width + 0.001 and extent.y1 <= height + 0.001, (name, extent)
    box = ax.get_position()
    area = box.width * box.height
    minimum_area = 0.62 if preset == "pair" else 0.67
    assert area >= minimum_area, (name, area)
    aspect = (box.width * width) / (box.height * height)
    assert abs(aspect - (1 if preset == "pair" else 1.5)) < 1e-8
    legend = ax.get_legend()
    if legend is not None:
        lb = legend.get_window_extent(renderer)
        for curve in ax.lines:
            path = curve.get_path().transformed(curve.get_transform())
            pad = (curve.get_markersize() / 2 + 1) * fig.dpi / 72
            expanded = Bbox.from_extents(
                lb.x0 - pad, lb.y0 - pad, lb.x1 + pad, lb.y1 + pad
            )
            assert not path.intersects_bbox(expanded, filled=False), (
                name,
                "legend intersects data",
            )
        assert len(legend.get_texts()) == 5
    path = BUILD / name
    fig.savefig(path, metadata={"Creator": "Matplotlib PGF with pdfLaTeX"})
    plt.close(fig)


def finite_matrix(
    c, t, w, ell, u, G, cells, gram_error=0.0, coefficient_error=0.0, epsilon=EPS
):
    assert c.shape[0] == 2
    if not np.isfinite(G).all() or la.eigvalsh(G).min() <= 0:
        raise la.LinAlgError("matrix_gram_failure")
    zeta = np.sqrt(np.clip(t, ell, u)) / (np.clip(t, ell, u) + t)
    d = np.sqrt(np.einsum("ij,ij->j", c, G @ c))
    a = w * zeta
    T = float(a @ d)
    Ytri = T * ((c * (a / d)) @ c.T)
    Ytri = (Ytri + Ytri.T) / 2
    centers = ell + (np.arange(cells) + 0.5) * (u - ell) / cells
    vv = (c * w) @ (1 / (centers[:, None] + t[None, :])).T * np.sqrt(centers)[None, :]
    gamma = float(np.max(np.sum(vv * la.solve(Ytri, vv, assume_a="pos"), axis=0)))
    Y = gamma * Ytri
    Y = (Y + Y.T) / 2
    ti = [ivfloat(x) for x in t]
    wi = [ivfloat(x) for x in w]
    ci = [[ivfloat(x) for x in row] for row in c]
    li = ivfloat(ell)
    ui = ivfloat(u)
    Mv = iv.mpf(0)
    Lv = iv.mpf(0)
    for j in range(len(t)):
        nc = iv.sqrt(ci[0][j] ** 2 + ci[1][j] ** 2)
        clamp = ivfloat(float(np.clip(t[j], ell, u)))
        Mv += wi[j] * iv.sqrt(clamp) / (clamp + ti[j]) * nc
        Lv += wi[j] * nc / (2 * iv.sqrt(li) * (li + ti[j]))
    radius = (
        interval_upper((ui - li) / (2 * cells))
        + np.finfo(float).eps * max(abs(ell), abs(u)) * 4
    )
    delta_grid = interval_upper(
        2 * Mv * Lv * ivfloat(radius) + (Lv * ivfloat(radius)) ** 2
    )
    delta_coeff = interval_upper(
        2 * Mv * ivfloat(coefficient_error) + ivfloat(coefficient_error) ** 2
    )
    delta = interval_upper(ivfloat(delta_grid) + ivfloat(delta_coeff))
    lower = 0.0
    yi = [[ivfloat(x) for x in row] for row in Y]
    for xi in centers:
        xx = ivfloat(xi)
        fac = iv.sqrt(xx)
        v = [
            fac
            * sum((wi[j] * ci[k][j] / (xx + ti[j]) for j in range(len(t))), iv.mpf(0))
            for k in range(2)
        ]
        aa = yi[0][0] - v[0] ** 2
        dd = yi[1][1] - v[1] ** 2
        bb = yi[0][1] - v[0] * v[1]
        mineig = (aa + dd - iv.sqrt((aa - dd) ** 2 + 4 * bb**2)) / 2
        lower = min(lower, interval_lower(mineig))
    kappa = max(0.0, -lower)
    margin = 128 * np.finfo(float).eps * max(np.linalg.norm(Y), kappa + delta)
    Ycert = Y.copy()
    for k in range(2):
        Ycert[k, k] = np.nextafter(Y[k, k] + kappa + delta + margin, np.inf)
    obj = sum(
        (ivfloat(Ycert[i, j]) * ivfloat(G[j, i]) for i in range(2) for j in range(2)),
        iv.mpf(0),
    )
    obj += ivfloat(gram_error) * (ivfloat(Ycert[0, 0]) + ivfloat(Ycert[1, 1]))
    eta = interval_upper(iv.sqrt(obj / (1 - ivfloat(epsilon))))
    return dict(
        eta_matrix=eta,
        eta_triangle=T / np.sqrt(1 - epsilon),
        ratio=eta / (T / np.sqrt(1 - epsilon)),
        gamma=gamma,
        Delta=delta,
        kappa=kappa,
        cells=cells,
        center_lower=lower,
        gram_error=gram_error,
        coefficient_error=coefficient_error,
        interval_checked=True,
    )


def matrix_roundoff(Omega, G, mu, c, t, w, ell, u):
    e2 = iv.mpf(0)
    for k in range(2):
        for l in range(2):
            exact = sum(
                (ivfloat(row[k]) * ivfloat(row[l]) for row in Omega[:, 2:]), iv.mpf(0)
            )
            gap = exact - ivfloat(G[k, l])
            bound = max(abs(interval_lower(gap)), abs(interval_upper(gap)))
            e2 += ivfloat(bound) ** 2
    theta = interval_upper(iv.sqrt(e2))

    ev = iv.mpf(0)
    for j, tj in enumerate(t):
        tt = ivfloat(tj)
        den = (2 + tt) ** 2 - 1
        exact = [-ivfloat(mu) * (2 + tt) / den, ivfloat(mu) / den]
        sq = iv.mpf(0)
        for k in range(2):
            gap = exact[k] - ivfloat(c[k, j])
            bound = max(abs(interval_lower(gap)), abs(interval_upper(gap)))
            sq += ivfloat(bound) ** 2
        z = ivfloat(float(np.clip(tj, ell, u)))
        ev += ivfloat(w[j]) * iv.sqrt(z) / (z + tt) * iv.sqrt(sq)
    return theta, interval_upper(ev)


def rank_two_output_error(t, w, x, u):
    x1, x2 = iv.mpf(0), iv.mpf(0)
    for shift, weight in zip(t, w):
        diagonal = 2 + ivfloat(shift)
        w_scaled = ivfloat(weight) / (diagonal**2 - 1)
        x1 += diagonal * w_scaled
        x2 -= w_scaled
    err2 = (x1 - ivfloat(x[0])) ** 2 + (x2 - ivfloat(x[1])) ** 2
    err2 += sum(ivfloat(component) ** 2 for component in x[2:])
    return interval_upper(iv.sqrt(ivfloat(u) * err2))
