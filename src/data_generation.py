"""
Parameterized contractive cyclic nonlinear system family (Section 3.2 of the
paper), constructed so that the root is guaranteed unique via the Banach
fixed-point theorem (Section 3.4, Theorem 1).

Design:
    f_i(x) = x_i - eps * ( a_{i,1} sin(x_j) + a_{i,2} cos(x_k) ) - b_i
    j = (i+1) mod n,  k = (i+2) mod n
    a_{i,1}, a_{i,2} ~ U(0.5, 2.0),  b_i ~ U(-1, 1),  eps = 0.2

Rewriting F(x) = 0 as the fixed-point equation x = Phi(x), where
    Phi(x)_i = b_i + eps * (a_{i,1} sin(x_j) + a_{i,2} cos(x_k)),
Phi is a contraction on (R^n, ||.||_inf): for any i,
    |Phi(x)_i - Phi(y)_i| <= eps*(a_{i,1} + a_{i,2}) * ||x-y||_inf
                           <= eps * 4 * ||x-y||_inf = 0.8 ||x-y||_inf
since a_{i,1}, a_{i,2} <= 2. The Lipschitz constant 0.8 < 1, so by the Banach
fixed-point theorem there is a unique real solution x*, reachable from any
starting point by fixed-point iteration. The sin/cos coupling across
variables keeps the system genuinely nonlinear.
"""
import numpy as np

EPS = 0.2

def Phi(x, A, b):
    """Vectorized fixed-point map. x: (..., n), A: (..., n, 2), b: (..., n)."""
    n = x.shape[-1]
    out = np.zeros_like(x)
    for i in range(n):
        j = (i + 1) % n
        k = (i + 2) % n
        out[..., i] = b[..., i] + EPS * (A[..., i, 0]*np.sin(x[..., j]) + A[..., i, 1]*np.cos(x[..., k]))
    return out

def F_eval(x, A, b):
    """Residual F(x; A, b) = x - Phi(x; A, b)."""
    return x - Phi(x, A, b)

def generate_dataset(n, N, seed, iters=150):
    """Generate N instances of the dimension-n system and their unique
    reference solutions x*, computed by `iters` fixed-point iterations
    starting from the zero vector (Section 3.4, Equation 10)."""
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.5, 2.0, size=(N, n, 2))
    b = rng.uniform(-1.0, 1.0, size=(N, n))
    x = np.zeros((N, n))
    for _ in range(iters):
        x = Phi(x, A, b)
    x_true = x
    p = np.concatenate([A.reshape(N, -1), b], axis=1)
    return p.astype(np.float32), x_true.astype(np.float32), A.astype(np.float32), b.astype(np.float32)
