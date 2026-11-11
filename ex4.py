from decimal import Decimal

import numpy as np
from threadpoolctl import threadpool_limits

from common import BUILD, add_up, fixed, projection, sinc, write_table
from common import finite_matrix, matrix_roundoff, rank_two_output_error


def run():
    rows = []
    for mu in [0.125, 0.25, 0.5]:
        H = np.array([[2.0, 1.0], [1.0, 2.0]])
        A = np.block([[H, mu * np.eye(2)], [mu * np.eye(2), H]])
        b = np.eye(4)[:, 0]
        ell, u = 1 - mu, 3 + mu
        rule = sinc(0.5, ell, u, 1.0, 1e-5)
        identity = np.eye(4)
        obj = projection(
            A,
            b,
            identity[:, :2],
            identity,
            rule["t"],
            rule["w"],
            ell,
            u,
            identity,
            epsilon=0.0,
        )
        theta, coefferr = matrix_roundoff(
            np.eye(4), np.eye(2), mu, -mu * obj["ys"], rule["t"], rule["w"], ell, u
        )
        mat = finite_matrix(
            -mu * obj["ys"],
            rule["t"],
            rule["w"],
            ell,
            u,
            np.eye(2),
            512,
            theta,
            coefferr,
            epsilon=0.0,
        )
        output_error = rank_two_output_error(rule["t"], rule["w"], obj["x"], u)
        upper = add_up(mat["eta_matrix"], output_error)
        residual = obj["eta_res"]
        tau = rule["eta_sinc"] + (upper + residual) / 2
        full = projection(
            A,
            b,
            identity,
            identity,
            rule["t"],
            rule["w"],
            ell,
            u,
            identity,
            epsilon=0.0,
        )
        full_total = add_up(full["eta_res"], rule["eta_sinc"])
        assert (
            add_up(rule["eta_sinc"], upper) < tau < rule["eta_sinc"] + residual
            and full_total < tau
        )
        rows.append(
            dict(
                mu=mu,
                residual_bound=residual,
                matrix_bound=upper,
                tau=tau,
                residual_stop_dimension=4,
                matrix_stop_dimension=2,
                projected_output_error=output_error,
                eta_sinc=rule["eta_sinc"],
                full_space_total=full_total,
                full_space_executed=True,
                tolerance_selected_after_bounds=True,
            )
        )
    return rows


def main():
    BUILD.mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        bounds = run()
    rows = []
    for r in bounds:
        eta_sinc = Decimal(str(r["eta_sinc"]))
        residual = Decimal(str(r["residual_bound"])) + eta_sinc
        matrix = Decimal(str(r["matrix_bound"])) + eta_sinc
        rows.append(
            [
                fixed(r["mu"], 3),
                fixed(residual, upward=True),
                fixed(matrix, upward=True),
                fixed(r["tau"]),
            ]
        )
    write_table(4, ["mu", "eta_total", "eta_total_plus", "tau"], rows)


if __name__ == "__main__":
    main()
