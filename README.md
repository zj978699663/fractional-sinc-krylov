# Fractional sinc Krylov experiments

## Scope

This package reproduces Tables 1–5 and Figures 1–2 of *Certified finite sinc Krylov approximations with sketching and recycling for fractional inverse matrix actions*.

## Run

Use Python 3.13.9 and pdfLaTeX with Computer Modern, `fontenc`, and `amsmath`. Run from this directory.

```text
python -m pip install numpy==2.3.5 scipy==1.16.3 mpmath==1.3.0 matplotlib==3.10.6 threadpoolctl==3.5.0
python ex1.py
python ex2.py
python ex3.py
python ex4.py
python ex5.py
```

## Tables, figures and checks

Each command can run on its own. Results go to `build/`.

| Paper | Command | Calculation and checks | Output |
|---|---|---|---|
| §7.1 Adaptive validation along recycled sequences | `python ex1.py` | Generate the real and complex sequences. Recompute 64 candidates, certify the first accepted one at each stage, and plot the real paths. | `table1.csv`, `fig1a.pdf`, `fig1b.pdf` |
| §7.2 Finite sinc tolerance allocation | `python ex2.py` | Compute the finite rule and its energy error and bound for 18 spectral, exponent, and tolerance cases. | `table2.csv` |
| §7.3 Matrix refinement over the spectral interval | `python ex3.py` | Recompute the rank-two certificates for nine parameter pairs at 512 cells and the refinement curve at four cell counts. | `table3.csv`, `fig2.pdf` |
| §7.4 Earlier stopping with the matrix certificate | `python ex4.py` | Run both stopping candidates and check the crossing for the three couplings. | `table4.csv` |
| §7.5 Sparse sequences and recycling | `python ex5.py` | Build the sparse matrices and sketches, recompute the stopping dimensions for both validators, and check the sine-transform reference by banded resolvent quadrature. Sum the original stage times and report the median and range. | `table5.csv` |

The real and complex sequence seeds are `20260816` and `20260916`. For Table 5, the input and validation seeds are `20261011+n` and `20261012+n+stage`, with stages numbered from zero. The 108 original stage times are in `data/timings.csv`. Table checks use the values in `reference/`.
