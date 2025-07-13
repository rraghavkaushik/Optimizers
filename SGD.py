''' Implementation of Stochastic Gradient descent in PyTorch. '''

import torch
import torch.nn as nn

class StochasticGradientDescent:
    def __init__(self, model, loss_fn = nn.MSELoss(), lr = 0.01):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.losses = []

    def fit(self, X, y, epochs = 10000, verbose = True, get_loss = False):
        ''' Shape of X (assumed here for simplicity): (n_samples, n_features)'''
        n_samples = X.shape[0]
        for epoch in epochs:
            loss_per_epoch = 0.0
            for i in range(n_samples):
                random_sample_idx = torch.randint(0, n_samples, (1, )).item()


                xi = X[random_sample_idx: random_sample_idx + 1]
                yi = y[random_sample_idx: random_sample_idx + 1]
    
                y_pred = self.model(xi)
                loss = self.loss_fn(y_pred, yi)

                loss_per_epoch += loss.item()

                loss.backward()
                with torch.no_grad():
                    for paramter in self.model.parameters():
                        parameter -= self.lr * parameter.grad
                    
                self.model.zero_grad()

            avg_loss = loss_per_epoch / n_samples
            # print(type(avg_loss))
            self.losses.append(avg_loss)

            if verbose and epoch % 50 == 0:
                print(f'epoch {epoch}, loss: {avg_loss:.4f}')

        if get_loss:
            print(f'loss after trainig: {self.losses}')
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
        plt.plot(self.losses, color='purple', alpha=0.7, linewidth=2)
        plt.title('Random Sample SGD Loss Curve (single sample updates)', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        

