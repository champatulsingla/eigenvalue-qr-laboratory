import numpy as np
import matplotlib.pyplot as plt
import eigen_lab
import test

TOL, MAX_ITER = 1e-8, 10000

# Experiment 1: Matrix with a clearly dominant eigenvalue
A = np.diag([5.0, 2.5, 1.0])
x0 = np.array([1.0, 1.0, 1.0])

power_method = eigen_lab.PowerIteration(A)

eigenvalue, eigenvector, history = power_method.power_iteration(
    x0, tol=TOL, max_iter=MAX_ITER
)

reference_values = np.linalg.eigvals(A)
reference_value = reference_values[np.argmax(np.abs(reference_values))]

iterations = np.arange(1, len(history) + 1)
eigenvalue_estimates = np.array([item[0] for item in history])
residuals = np.array([item[1] for item in history])

print("\nExperiment 1: Clearly dominant eigenvalue")
print("NumPy reference eigenvalue:", reference_value)
print("Power Iteration eigenvalue:", eigenvalue)
print("Number of iterations:      ", len(history))
print("Final residual:            ", residuals[-1])

plt.figure(figsize=(11, 4))

plt.subplot(1, 2, 1)
plt.plot(iterations, eigenvalue_estimates, marker=".", label="Power Iteration")
plt.axhline(reference_value, color="red", linestyle="--", label="NumPy reference")
plt.xlabel("Iteration")
plt.ylabel("Eigenvalue estimate")
plt.title("Eigenvalue convergence")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.semilogy(iterations, residuals, marker=".")
plt.xlabel("Iteration")
plt.ylabel("Residual norm")
plt.title("Residual convergence")
plt.grid(True)

plt.tight_layout()
plt.savefig("power_iteration_convergence.png", dpi=300)
