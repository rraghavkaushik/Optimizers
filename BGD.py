'''Code for Batch gradient Descent or Vanilla Gradient Descent'''

import torch
import torch.nn as nn

class BatchGradientDescent:
    def __init__(self, model, loss_fn = nn.MSELoss(), lr = 0.01):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.losses = []

    def fit(self, X, y, epochs = 1000, verbose = True):

        for epoch in epochs:
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)

            self.losses.append(loss.items())

            loss.backward()

            with torch.no_grad():
                for parameter in self.model.parameters():
                    parameter -= self.lr * paramter.grad

            self.model.zero_grad()

        if verbose and epoch % 10 == 0:
            print(f'epoch {epoch}, loss: {loss.item():.4f}')

        return self
    
    def get_losses(self):
        return self.losses
    
    def plot_loss_curves(self):
        import matplotlib.pyplot as plt
        
        if not self.losses:
            print("No loss history found. Train the model first.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses, color='blue', alpha=0.7, linewidth=2)
        plt.title('Batch Gradient Descent Loss Curve', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
