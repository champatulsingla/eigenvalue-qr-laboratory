# eigenlab

A small NumPy library for eigenvalue computations: power iteration, Gram-Schmidt QR factorization, and the unshifted QR iteration, plus a `Testing` class that runs ready-made experiments.

## Importing

```python
from eigenlab import PowerIteration, QRFactorization, QR_Iteration, Testing
```

You can also import the module that holds the first three classes:

```python
from eigenlab import eigen_lab
pi = eigen_lab.PowerIteration(A)
```

Importing the package loads NumPy, SciPy and Matplotlib, so all three must be available. The examples below also use `import numpy as np`.

| Class | Purpose |
|---|---|
| [`PowerIteration`](#poweriteration) | Dominant eigenvalue and eigenvector of a square matrix |
| [`QRFactorization`](#qrfactorization) | QR factorization by classical or modified Gram-Schmidt, with error measures |
| [`QR_Iteration`](#qr_iteration) | All eigenvalues of a square matrix by repeated QR factorization, with a convergence plot |
| [`Testing`](#testing) | Ready-made experiments for the three classes above |

---

## `PowerIteration`

Finds the eigenvalue of largest magnitude (the dominant eigenvalue) and a matching unit eigenvector.

### `PowerIteration(A)`

| | |
|---|---|
| **Input** | `A`: a square matrix as a **NumPy array** of shape `(n, n)` (real or complex). |
| **Output** | A `PowerIteration` object. |
| **Raises** | `ValueError("Matrix A must be a square matrix.")` if `A` is not 2-D or not square. |
| | `TypeError` if `A` is a plain Python list. Convert it first with `np.array(...)`. |

### `power_iteration(x0, tol=1e-8, max_iter=10000)`

| | |
|---|---|
| **Input** | `x0`: starting vector as a **NumPy array** of shape `(n,)`. It is normalized internally. It must have a nonzero component along the dominant eigenvector. |
| | `tol` (float, default `1e-8`): stop when the residual `‖A x - λ x‖` falls below `tol`. |
| | `max_iter` (int, default `10000`): maximum number of iterations. Use at least `1`. |
| **Output** | `(eigenvalue, eigenvector, history)` |
| | `eigenvalue`: the estimate (Rayleigh quotient). It is complex if `A` is complex. |
| | `eigenvector`: unit-norm array of shape `(n,)`. Its sign (or phase) is not fixed. |
| | `history`: list of `(eigenvalue, residual)` tuples, one per iteration, so `len(history)` is the number of iterations used. |
| **Raises** | `ValueError("Initial vector x0 must have the same dimension as the number of columns in matrix A.")` if `x0` is not 1-D of length `n`. |
| | `ValueError("Initial vector x0 must not be the zero vector.")` if `x0` is all zeros. |
| | `RuntimeError("Power iteration did not converge within ... iterations. Final residual = ...")` if the residual never drops below `tol`. |
| | `TypeError` if `x0` is a plain Python list. |
| | `UnboundLocalError` if `max_iter=0`. |
| **Warns** | `UserWarning` if `A x` becomes (numerically) zero. The method then returns `(0.0, x, [(0.0, 0.0)])`. The message says whether `x0` lies in the null space of a non-zero `A` (try another `x0`) or `A` is effectively the zero matrix. |

```python
A = np.array([[2., 1.],
              [1., 3.]])
pi = PowerIteration(A)
value, vector, history = pi.power_iteration(np.array([1., 1.]))

print(value)          # 3.618...
print(vector)         # unit eigenvector
print(len(history))   # iterations used
```

**When it does not converge.** Power iteration needs a single dominant eigenvalue. It raises `RuntimeError` when two eigenvalues share the largest magnitude (for example `+1` and `-1` for `[[0, 1], [1, 0]]`) or when the dominant eigenvalues are a complex pair (for example a rotation matrix). If `x0` has no component along the dominant eigenvector, the method converges to a different eigenvalue. For `diag(5, 2)` with `x0 = [0, 1]` it returns `2.0`.

---

## `QRFactorization`

Computes the QR factorization `A = QR` of a matrix using Gram-Schmidt methods, and provides error measures for checking the result.

- `Q` has shape `(m, n)` with orthonormal columns.
- `R` has shape `(n, n)` and is upper triangular.

### `QRFactorization(A)`

| | |
|---|---|
| **Input** | `A`: a 2-D array-like of numbers (nested list or NumPy array) with `m` rows and `n` columns. It is converted to a float array internally and the original is not modified. |
| **Output** | A `QRFactorization` object. |
| **Raises** | `ValueError("Please enter 2 dimensional matrix only.")` if `A` is not 2-D (for example a flat list or a 3-D array). |
| | `ValueError` (from NumPy) if `A` cannot be converted to a float array, such as ragged rows or non-numeric entries. |

```python
qr = QRFactorization([[1, 1], [1, 0], [0, 1]])
```

### `gramschmidt(tol=1e-8)`

Computes `Q` and `R` with **classical** Gram-Schmidt.

| | |
|---|---|
| **Input** | `tol` (float, default `1e-8`): a column whose component left after removing projections onto earlier columns has norm below `tol` is treated as linearly dependent. The threshold is absolute, not relative to the size of the column. |
| **Output** | `(Q, R)`: a tuple of two NumPy arrays, `Q` of shape `(m, n)` and `R` of shape `(n, n)`. `R` is exactly upper triangular. |
| **Raises** | `ValueError("Number of columns must be less than or equal the number of rows.")` if `n > m` (a "fat" matrix). |
| | `ValueError("Matrix has linearly dependent or almost linearly dependent columns")` if any column is dependent on earlier columns within `tol`. |

```python
Q, R = qr.gramschmidt()
Q, R = qr.gramschmidt(tol=1e-12)    # accept columns closer to dependent
```

### `modified_gramschmidt(reorthogonalize=False, tol=1e-8)`

Computes `Q` and `R` with **modified** Gram-Schmidt, optionally followed by a second orthogonalization pass.

| | |
|---|---|
| **Input** | `reorthogonalize` (bool, default `False`): if `True`, the method checks the orthogonality error of the first-pass `Q`. If it is greater than `tol`, a second pass is run and the factors are combined so that `A = QR` still holds. If the error is already at most `tol`, the second pass is skipped. |
| | `tol` (float, default `1e-8`): used for **two** purposes, the linear-dependence threshold (as in `gramschmidt`) and the trigger for the second pass. It is an absolute threshold. |
| **Output** | `(Q, R)`: a tuple of two NumPy arrays, `Q` of shape `(m, n)` and `R` of shape `(n, n)` (upper triangular). |
| **Raises** | `ValueError("Number of columns must be less than or equal the number of rows.")` if `n > m`. |
| | `ValueError("Matrix has linearly dependent or almost linearly dependent columns")` if a column is dependent within `tol`. This check happens in the first pass, so it is raised even when `reorthogonalize=True`. |

```python
Q, R = qr.modified_gramschmidt()
Q, R = qr.modified_gramschmidt(reorthogonalize=True)
Q, R = qr.modified_gramschmidt(reorthogonalize=True, tol=1e-12)
```

`reorthogonalize` exists only on `modified_gramschmidt`. Passing it to `gramschmidt` raises a `TypeError`.

### `reconstruction(q, r)`

| | |
|---|---|
| **Input** | `q`: array of shape `(m, n)`. `r`: array of shape `(n, n)`. |
| **Output** | float: the Frobenius norm `‖A - QR‖`. |
| **Raises** | `ValueError` (from NumPy) if the shapes of `q`, `r` and `A` are incompatible. |

### `orthogonality(q)`

| | |
|---|---|
| **Input** | `q`: array of shape `(m, n)` where `n` is the number of columns of `A`. |
| **Output** | float: the Frobenius norm `‖QᵀQ - I‖`. It is close to machine precision (about 1e-16) for a good `Q`. |
| **Raises** | `ValueError` (from NumPy) if `q` does not have `n` columns. |

### `triangularity(q)`

Measures the part of `A` that "leaks" onto later columns of `Q`. Despite the name, it takes **`Q`, not `R`**.

| | |
|---|---|
| **Input** | `q`: array of shape `(m, n)`, computed from the same matrix `A` as this object. |
| **Output** | float: the Frobenius norm of the strictly lower triangle of `QᵀA`, that is `‖tril(QᵀA, -1)‖`. It is zero in exact arithmetic. |
| **Raises** | `ValueError` (from NumPy) if the shapes of `q` and `A` are incompatible. |

### Full example

```python
A = np.array([[1., 1., 0.],
              [1., 0., 1.],
              [0., 1., 1.],
              [1., 1., 1.]])

qr = QRFactorization(A)
Q, R = qr.modified_gramschmidt(reorthogonalize=True)

print(qr.reconstruction(Q, R))   # ||A - QR||
print(qr.orthogonality(Q))       # ||Q^T Q - I||
print(qr.triangularity(Q))       # ||tril(Q^T A, -1)||

try:
    QRFactorization([[1, 2], [2, 4], [3, 6]]).modified_gramschmidt()
except ValueError as e:
    print(e)    # Matrix has linearly dependent or almost linearly dependent columns
```

---

## `QR_Iteration`

Finds all eigenvalues of a square matrix by repeating `A <- R Q`, where `A = Q R` is computed with modified Gram-Schmidt (with reorthogonalization). When `A` becomes upper triangular, its diagonal holds the eigenvalues.

### `QR_Iteration(A, max_iter, tol)`

All three arguments are required.

| | |
|---|---|
| **Input** | `A`: square matrix (nested list or NumPy array), shape `(n, n)`. |
| | `max_iter` (int): maximum number of QR steps. |
| | `tol` (float): the iteration stops when the norm of the strictly lower triangle of `A` is below `tol`. The same value is also passed to `modified_gramschmidt` as its dependence threshold and reorthogonalization trigger. |
| **Output** | A `QR_Iteration` object. NumPy's eigenvalues of `A` are computed once for comparison in the plot. |
| **Raises** | `ValueError("Expecting a 2D square matrix")` if `A` is not 2-D and square. |
| | `TypeError` if `max_iter` or `tol` is missing. |

**Attributes** after construction and use:

| Attribute | Meaning |
|---|---|
| `A` | The current iterate (starts as a float copy of the input and is overwritten at each step). Its diagonal holds the eigenvalue estimates. |
| `n` | Matrix size. |
| `count` | Number of QR steps taken in the most recent call to `iterate()`. |
| `converged` | `True` if the last call to `iterate()` ended with a small lower triangle. |
| `diag_history` | List with one entry per iteration: the diagonal of `A`, sorted by decreasing magnitude. |
| `np_eigs` | NumPy's eigenvalues of the original matrix, sorted by decreasing magnitude. |

### `stopping_Check()`

| | |
|---|---|
| **Input** | None. |
| **Output** | `True` if the Frobenius norm of the strictly lower triangle of the current `A` is below `tol`, otherwise `False`. |

### `iterate()`

| | |
|---|---|
| **Input** | None. |
| **Output** | `True` if the iteration converged within `max_iter` steps, otherwise `False`. Not converging does **not** raise an error. Check `converged` or the return value. |
| **Raises** | `ValueError("Matrix has linearly dependent or almost linearly dependent columns")` if a QR step meets a (nearly) dependent column. This happens for singular matrices, such as `[[1, 2], [2, 4]]`. |

Calling `iterate()` again continues from the current `A` for up to `max_iter` more steps and appends to `diag_history`. If `A` is already upper triangular, `iterate()` returns `True` immediately with `count = 0` and no history.

```python
A = np.array([[4., 1.],
              [2., 3.]])
qr_it = QR_Iteration(A, max_iter=200, tol=1e-8)

if qr_it.iterate():
    print(np.diag(qr_it.A))      # [5. 2.]  eigenvalue estimates
    print(qr_it.count)           # number of QR steps
    print(qr_it.np_eigs)         # NumPy's values, for comparison
```

### `plot_convergence()`

| | |
|---|---|
| **Input** | None. Call `iterate()` first. |
| **Output** | `None`. It shows a Matplotlib figure with one panel per eigenvalue: the diagonal entry against the iteration number, with a dashed line at NumPy's eigenvalue. |
| **Behavior** | If there is no history yet, it prints ` Please call iterate first ` and returns without plotting. |

**Limits of the method.** This is the plain (unshifted) QR iteration. It converges when the eigenvalues are real and have distinct magnitudes. For matrices with complex eigenvalues (for example a rotation matrix), `iterate()` returns `False`, and `plot_convergence()` draws only the real parts and Matplotlib emits a `ComplexWarning`.

---

## `Testing`

Runs experiments on the classes above and reports the results.

### `Testing(random_state=2026)`

| | |
|---|---|
| **Input** | `random_state` (int, default `2026`): seed for random test matrices. |
| **Output** | A `Testing` object. |

### `testQR_Factorization(A=None, dim=(5,3,3), method='modified_gramschmidt', reorthogonalize=False, tol=1e-12)`

Factors every matrix in a stack with the chosen method and returns the factors and errors.

| | |
|---|---|
| **Input** | `A`: a stack of matrices, a 3-D array-like of shape `(count, m, n)`. If `None`, `count` random matrices with entries uniform in `[0, 1)` are generated from `random_state`. |
| | `dim`: tuple `(count, m, n)` giving the number and size of the random matrices. Used only when `A` is `None`. |
| | `method`: `'gramschmidt'` or `'modified_gramschmidt'`. |
| | `reorthogonalize`: passed to `modified_gramschmidt`. Ignored when `method='gramschmidt'`. |
| | `tol`: passed to the chosen method. Note that the default here is `1e-12`, not `1e-8`. |
| **Output** | A tuple of five items, in this order: |
| | 1. `Q`: list of `count` arrays |
| | 2. `R`: list of `count` arrays |
| | 3. `reconstruction_errors`: array of length `count` |
| | 4. `orthogonality_errors`: array of length `count` |
| | 5. `triangularity_errors`: array of length `count` |
| **Raises** | `ValueError("Invalid method")` if `method` is not one of the two names above. |
| | `ValueError("Please enter 2 dimensional matrix only.")` if `A` is a single 2-D matrix instead of a stack. Wrap a single matrix as `np.array([M])`. |
| | Any `ValueError` from the factorization (fat matrix or linearly dependent columns). It is not caught, so the whole call fails. |

```python
t = Testing()

Q, R, rec, orth, tri = t.testQR_Factorization()                  # 5 random 3x3, modified GS
Q, R, rec, orth, tri = t.testQR_Factorization(dim=(20, 6, 6), method='gramschmidt')

M = np.array([[1., 1.], [1., 0.], [0., 1.]])
Q, R, rec, orth, tri = t.testQR_Factorization(A=np.array([M]), reorthogonalize=True)
```

### `test_PowerMethod(A=None, x0=None, tol=1e-8, max_iter=10000)`

Runs `PowerIteration` on a matrix, compares the result with NumPy, and prints a summary.

| | |
|---|---|
| **Input** | `A`: square **NumPy array**. Default: `[[5, 0], [0, 2]]`. |
| | `x0`: starting vector as a **NumPy array**. Default: `[1, 1]`. |
| | `tol`, `max_iter`: passed to `power_iteration`. |
| **Output** | Tuple `(value, vector, history, residual, passed)`. `value`, `vector` and `history` are as in `power_iteration`. `residual` is `‖A v - λ v‖` for the returned pair. `passed` is `True` if `residual < tol` and the eigenvalue differs from NumPy's dominant eigenvalue by less than `10 * tol`. |
| **Prints** | The computed and reference eigenvalue, the eigenvector, the iteration count, the residual, the eigenvalue error, and `Test result: PASS` or `FAIL`. |
| **Raises** | Whatever `PowerIteration` raises (`ValueError` for a non-square `A` or bad `x0`, `RuntimeError` if it does not converge, `TypeError` for list inputs). Nothing is caught. |

```python
value, vector, history, residual, passed = Testing().test_PowerMethod()
Testing().test_PowerMethod(A=np.array([[2., 1.], [1., 3.]]), x0=np.array([1., 0.]))
```

### `testQR_iteration(A, max_iter, tol)`

Runs `QR_Iteration` on a matrix, prints whether it converged, and shows the convergence plot. All three arguments are required.

| | |
|---|---|
| **Input** | `A`: square matrix. `max_iter`: maximum QR steps. `tol`: stopping tolerance. |
| **Output** | `None`. |
| **Prints** | `convergence is :  True` or `False`, followed by the plot. If a QR step fails, it prints `QR iteration stopped` and the error message instead. |
| **Raises** | `ValueError("Expecting a 2D square matrix")` if `A` is not square. It is raised when the `QR_Iteration` object is built, outside the `try`, so it is **not** caught. |
| | `TypeError` if an argument is missing. |

A `ValueError` from the iteration itself (linearly dependent columns, e.g. a singular matrix) is caught and printed.

```python
Testing().testQR_iteration(np.array([[4., 1.], [2., 3.]]), max_iter=200, tol=1e-8)
```

---

## Summary of errors and warnings

| Raised | By | Cause |
|---|---|---|
| `ValueError("Matrix A must be a square matrix.")` | `PowerIteration(A)` | `A` not 2-D or not square |
| `ValueError("Initial vector x0 must have the same dimension ...")` | `power_iteration` | `x0` not 1-D of length `n` |
| `ValueError("Initial vector x0 must not be the zero vector.")` | `power_iteration` | `x0` is all zeros |
| `RuntimeError("Power iteration did not converge ...")` | `power_iteration` | residual not below `tol` within `max_iter` |
| `UserWarning` | `power_iteration` | `A x` became zero (`x0` in the null space, or `A` is zero) |
| `TypeError` | `PowerIteration`, `power_iteration`, `test_PowerMethod` | `A` or `x0` is a plain list |
| `UnboundLocalError` | `power_iteration` | `max_iter=0` |
| `ValueError("Please enter 2 dimensional matrix only.")` | `QRFactorization(A)`, `testQR_Factorization` | `A` not 2-D (or a single matrix passed instead of a stack) |
| `ValueError("Number of columns must be less than or equal the number of rows.")` | `gramschmidt`, `modified_gramschmidt` | more columns than rows |
| `ValueError("Matrix has linearly dependent or almost linearly dependent columns")` | `gramschmidt`, `modified_gramschmidt`, `QR_Iteration.iterate` | a column is dependent on earlier columns within `tol` |
| `ValueError("Expecting a 2D square matrix")` | `QR_Iteration(A, ...)`, `testQR_iteration` | `A` not 2-D and square |
| `ValueError("Invalid method")` | `testQR_Factorization` | `method` not `'gramschmidt'` or `'modified_gramschmidt'` |
| `ValueError` (NumPy) | `QRFactorization` and its error measures | input not convertible to floats, or incompatible shapes |
| `TypeError` | `gramschmidt` | `reorthogonalize` passed to it |
| `TypeError` | `QR_Iteration`, `testQR_iteration` | missing required argument |
