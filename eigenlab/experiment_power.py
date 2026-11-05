import numpy as np
import matplotlib.pyplot as plt
import eigen_lab
import test

if __name__=="__main__":

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
    
    
    # Experiment 2: Effect of the spectral ratio on convergence
    ratios = [0.50, 0.90, 0.99]
    iteration_counts = []
    
    plt.figure(figsize=(7, 5))
    
    for ratio in ratios:
        A = np.diag([1.0, ratio])
        x0 = np.array([1.0, 1.0])
    
        eigenvalue, eigenvector, history = eigen_lab.PowerIteration(A).power_iteration(
            x0, tol=TOL, max_iter=MAX_ITER
        )
    
        residuals = [item[1] for item in history]
        iteration_counts.append(len(history))
    
        plt.semilogy(
            range(1, len(history) + 1), residuals,
            label=f"Spectral ratio = {ratio:.2f}"
        )
        print(f"Spectral ratio {ratio:.2f}: {len(history)} iterations")
    
    plt.xlabel("Iteration")
    plt.ylabel("Residual norm")
    plt.title("Effect of spectral ratio on convergence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("power_iteration_spectral_ratio.png", dpi=300)
    
    # Experiment 3: Run the selected special-case checks from test.py
    # This checks spectral ratios, complex conjugates, a defective matrix,
    # and an initial vector that is annihilated by the matrix.
    print("\nExperiment 3: Special-case checks from test.py")
    tester = test.Testing()
    tests_passed = tester.test_PowerMethod(tol=TOL, max_iter= MAX_ITER)
    
    print("\nAll experiment plots saved.")
    print("1. power_iteration_convergence.png")
    print("2. power_iteration_spectral_ratio.png")
    print("Special-case tests:", "PASS" if tests_passed else "CHECK FAILED TESTS")
    
    plt.show()
