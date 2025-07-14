''' Implementation of Stochastic Gradient Descent with Momentum.
We define velocity vector v_t for parameter updates.

Update rules:
v_t = γ * v_t−1 + η * ∇θJ(θ)
θ = θ − v_t

where, γ - momentum (usually set to 0.9 by default)
η - Learning rate
'''

import torch
import torch.nn as nn
# import math

class SGDMomentum:
    def __init__(self, model, loss_fn = nn.MSELoss(), lr = 0.01, momentum = 0.9, batch_size = 32):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.losses = []
        self.vel = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.vel[name] = torch.zeros_like(parameter.data)
        self.batch_size = batch_size

    def fit(self, X, y, epochs = 1000, verbose = True):
        ''' assumption: X has the shape (batch_size, n_features)'''

        self.losses = []
        n_samples = X.shape[0]

        for epoch in range(epochs):
            loss_per_epoch = 0
            random_indices = torch.randperm(n_samples)

            X_shuffled = X[random_indices]
            y_shuffled = y[random_indices]

            for i in range(0, n_samples, self.batch_size):
                if i + self.batch_size > n_samples:
                    x_batch = X_shuffled[i : self.batch_size]
                    y_batch = y_shuffled[i : self.batch_size]
                else:
                    x_batch = X_shuffled[i : i + self.batch_size]
                    y_batch = y_shuffled[i : i + self.batch_size]           

                # print(x_batch, y_batch)    
                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_pred, y_batch)
                loss.backward()

                loss_per_epoch += loss.item()

                for name, parameter in self.model.named_parameters():
                    if parameter.requires_grad and parameter is not None:
                        self.vel[name] = self.momentum * self.vel[name] + self.lr * parameter.grad
                        parameter.grad -= self.lr * self.vel[name]

                self.model.zero_grad()
            
            # n_samples = math.ceil(n_samples / self.batch_size)
            n_batches = (n_samples + self.batch_size - 1) // self.batch_size
            avg_loss = loss_per_epoch / self.batch_size

            self.losses.append(avg_loss)

        if verbose and epoch % 50 == 0:
            print(f'epoch {epoch}, loss: {avg_loss:.4f}')

        return self
    
    def predict(self, X):
        with torch.no_grad():
            return self.model(X)
        
    def plot_loss_curves(self):
        import matplotlib.pyplot as plt
        
        if not self.losses:
            print("No loss history found. Train the model first.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses, color='blue', alpha=0.7, linewidth=2)
        plt.title(f'SGD with Momentum Loss Curve (momentum={self.momentum})', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()