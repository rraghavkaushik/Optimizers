''' Implementation of Stochastic Gradient Descent with Nesterov Accelerated Gradient(NAG).
NAG is a way to give us an approximation of the next position of the parameters. We can now effectively look ahead
by calculating the gradient not w.r.t. to our current parameters θ but w.r.t. the approximate future
position of our parameters. (An overview of gradient descent optimization algorithms (2017), S. Ruder.)

Nesterov Update Rules:
v_t = γ * v_t−1 + η * ∇θJ(θ − γ * v_t−1)
θ = θ − v_t

where, γ - momentum (usually set to 0.9 by default)
η - Learning rate
'''

import torch
import torch.nn as nn

class SGDNesterov:
    def __init__(self, model, loss_fn = nn.MSELoss(), lr = 0.01, momentum = 0.9, batch_size = 32):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.momentum = momentum
        self.losses = []

        self.vel = {}
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.vel[name] = torch.zeros_like(parameter.data)
        self.batch_size = batch_size

    def fit(self, X, y, epochs = 1000, verbose = True):
        self.losses = []
        n_samples = X.shape[0]

        for epoch in range(epochs):
            loss_per_epoch = 0

            random_indices = torch.randperm(n_samples)
            X_shuffled = X[random_indices]
            y_shuffled = y[random_indices]

            for i in range(0, n_samples, self.batch_size):
                if i + self.batch_size > n_samples:
                    x_batch = X_shuffled[i : n_samples]
                    y_batch = y_shuffled[i : n_samples]
                else:
                    x_batch = X_shuffled[i : i + self.batch_size]
                    y_batch = y_shuffled[i : i + self.batch_size]

                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_pred, y_batch)
                
                loss_per_epoch += loss.item()
                loss.backward()

                with torch.no_grad():
                    for name, parameter in self.model.named_parameters():
                        if parameter.requires_grad and parameter is not None:
                            self.parameter -= self.momentum * self.vel[name]
                            self.vel[name] = self.momentum * self.vel[name] + self.lr * parameter.grad
                            self.parameter -= self.vel[name]

                self.model.zero_grad()

            n_batches = (n_samples + self.batch_size - 1) // self.batch_size
            avg_loss = loss_per_epoch / n_batches
            self.losses.append(avg_loss)

            if verbose and epoch % 50 == 0:
                print(f'epoch {epoch}, loss: {avg_loss:.4f}')
        return self

    def get_losses(self):
        return self.losses
    
    def predot(self, X):
        with torch.no_grad():
            return self.model(X)
        
    def plot_losses(self):
        import matplotlib.pyplot as plt
        if not self.losses:
            print("No loss history found. Train the model first.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses, color='blue', alpha=0.7, linewidth=2)
        plt.title(f'SGD with NAG Loss Curve (momentum={self.momentum})', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()