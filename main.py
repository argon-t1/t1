"""
Deep Learning & Reverse-Mode Autograd from First Principles
-----------------------------------------------------------
Features implemented from scratch:
1. Scalar DAG Engine (Value): dynamically tracks computational graphs and
   evaluates gradients using reverse-mode autodiff with iterative topological sort.
2. Neural Network Primitives: Neuron, Layer, and MLP architectures.
3. Adam Optimizer: first/second moment estimates with bias correction.
4. Non-Linear Classification: trains on a concentric circles dataset.
5. ASCII Field Visualizer: renders the continuous decision boundary in the terminal.
"""

import math
import random
import time


# =====================================================================
# 1. Autograd Core: Computational Graph Node & Differentiation Engine
# =====================================================================

class Value:
    """Stores a scalar value and its accumulated gradient across a DAG."""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Power must be int or float"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other**-1)

    def __rtruediv__(self, other):
        return Value(other) * (self**-1)

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        # Clamped to prevent numerical exponent instability
        clamped = max(-20.0, min(20.0, self.data))
        t = math.tanh(clamped)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """Iterative topological sort to execute reverse-mode autodiff without recursion depth limits."""
        topo = []
        visited = set()
        stack = [(self, False)]

        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
            elif node not in visited:
                visited.add(node)
                stack.append((node, True))
                for child in node._prev:
                    if child not in visited:
                        stack.append((child, False))

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


# =====================================================================
# 2. Neural Network Primitives
# =====================================================================

class Neuron:
    def __init__(self, nin, nonlin=True):
        # Xavier/Glorot initialization scaled by fan-in
        bound = math.sqrt(6.0 / (nin + 1))
        self.w = [Value(random.uniform(-bound, bound)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        # w^T * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=(i != len(nouts) - 1))
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# =====================================================================
# 3. First-Principles Adam Optimizer
# =====================================================================

class Adam:
    def __init__(self, params, lr=0.08, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0] * len(params)
        self.v = [0.0] * len(params)
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            # First and second moment running averages
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)

            # Bias correction
            m_hat = self.m[i] / (1.0 - (self.beta1 ** self.t))
            v_hat = self.v[i] / (1.0 - (self.beta2 ** self.t))

            # Weight update
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0


# =====================================================================
# 4. Synthetic Concentric Rings Dataset
# =====================================================================

def generate_rings_dataset(num_samples=70, noise=0.05, seed=42):
    """Generates two non-linearly separable concentric circles."""
    random.seed(seed)
    X, y = [], []
    half = num_samples // 2

    # Inner circle: class +1
    for _ in range(half):
        theta = random.uniform(0, 2 * math.pi)
        r = 0.45 + random.gauss(0, noise)
        X.append([r * math.cos(theta), r * math.sin(theta)])
        y.append(1.0)

    # Outer circle: class -1
    for _ in range(half):
        theta = random.uniform(0, 2 * math.pi)
        r = 1.10 + random.gauss(0, noise)
        X.append([r * math.cos(theta), r * math.sin(theta)])
        y.append(-1.0)

    return X, y


# =====================================================================
# 5. Continuous Decision Boundary Terminal Renderer
# =====================================================================

def render_ascii_decision_boundary(model, X, y, width=61, height=25):
    """Samples the 2D plane and projects the learned contour to ASCII."""
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.3, 1.3
    dx = (x_max - x_min) / (width - 1)
    dy = (y_max - y_min) / (height - 1)

    canvas = []
    canvas.append("┌" + "─" * width + "┐")
    data_points = list(zip(X, y))

    for row in range(height):
        py = y_max - row * dy
        row_str = []
        for col in range(width):
            px = x_min + col * dx

            # Check if an actual data point lies within this pixel cell
            matched_label = None
            min_dist = float("inf")
            for (gx, gy), label in data_points:
                dist = ((gx - px) / dx) ** 2 + ((gy - py) / (dy * 1.6)) ** 2
                if dist < min_dist:
                    min_dist = dist
                    matched_label = label

            if min_dist <= 1.0:
                row_str.append("●" if matched_label > 0 else "▲")
            else:
                pred = model([px, py]).data
                if abs(pred) < 0.12:
                    row_str.append(":")  # Contour zero-crossing
                elif pred > 0:
                    row_str.append("·")  # Positive classification territory
                else:
                    row_str.append(" ")  # Negative classification territory

        canvas.append("│" + "".join(row_str) + "│")

    canvas.append("└" + "─" * width + "┘")
    return "\n".join(canvas)


# =====================================================================
# 6. Execution Pipeline
# =====================================================================

def main():
    print("=" * 63)
    print("  AUTOGRAD & NEURAL NETWORK ENGINE (STANDARD LIBRARY ONLY)")
    print("=" * 63)

    # 1. Generate data
    X, y = generate_rings_dataset(num_samples=70, seed=42)
    print(f"Generated {len(X)} samples across 2 concentric rings.")

    # 2. Build MLP: 2 Inputs -> 12 Hidden (tanh) -> 12 Hidden (tanh) -> 1 Output (linear)
    model = MLP(2, [12, 12, 1])
    optimizer = Adam(model.parameters(), lr=0.08)
    print(f"Model compiled with {len(model.parameters())} trainable parameters.")
    print("\nTraining Progress:")
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Accuracy':>10} | {'Grad Norm':>10}")
    print("-" * 44)

    start_time = time.time()
    epochs = 60

    # 3. Optimization loop
    for epoch in range(epochs + 1):
        # Forward pass
        y_preds = [model(x) for x in X]

        # Mean Squared Error Loss
        loss_data = sum((yp - yt) ** 2 for yp, yt in zip(y_preds, y)) * (1.0 / len(y))

        # L2 Regularization (Weight decay)
        reg_loss = sum(p * p for p in model.parameters()) * 1e-4
        total_loss = loss_data + reg_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Step
        optimizer.step()

        # Logging
        if epoch % 10 == 0:
            correct = sum((yp.data > 0) == (yt > 0) for yp, yt in zip(y_preds, y))
            acc = correct / len(y)
            grad_norm = math.sqrt(sum(p.grad**2 for p in model.parameters()))
            print(f"{epoch:>6} | {total_loss.data:>8.4f} | {acc * 100:>9.1f}% | {grad_norm:>10.4f}")

    elapsed = time.time() - start_time
    print("-" * 44)
    print(f"Training completed in {elapsed:.2f} seconds.")

    # 4. Render learned decision surface
    print("\nLearned Continuous Decision Boundary:")
    print("Legend:  ● Class +1 (Inner)   ▲ Class -1 (Outer)   : Decision Boundary")
    print(render_ascii_decision_boundary(model, X, y))

    # 5. Direct inference test
    test_points = [[0.0, 0.0], [0.4, 0.0], [1.1, 0.0], [0.0, 1.2]]
    print("\nPoint Inference Checks:")
    for pt in test_points:
        score = model(pt).data
        predicted_class = "+1 (Inner)" if score > 0 else "-1 (Outer)"
        print(f" Point ({pt[0]:>4.1f}, {pt[1]:>4.1f}) -> Score: {score:>7.3f} => {predicted_class}")


if __name__ == "__main__":
    main()
