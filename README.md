# Reverse-mode automatic differentiation
This standalone Python script implements a complete reverse-mode automatic differentiation (autograd) engine with dynamic DAG computation graphs, and uses it to construct and train a multi-layer perceptron (MLP) using a custom Adam optimizer from scratch to solve a non-linearly separable classification problem—complete with terminal ASCII decision-boundary rendering.

It requires no third-party dependencies (no PyTorch, NumPy, or SciPy) and runs entirely on the Python standard library.
