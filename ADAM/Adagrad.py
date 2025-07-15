import torch
import torch.nn as nn

class Adagrad:
    def __init__(self, model, loss_fn = nn.MSELoss(), lr = 0.01, batch_size = 32, epsilon = 1e-8):
        self.model = model
        self.loss_fn = loss_fn
        self.lr= lr
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.losses = []
        self.G = {}
        for name, parameter in model.named_parameters():
            if parameter.requiers_grad:
                self.G[name] = torch.zeros_like(parameter.data)

    def fit(self, X, y, epochs =1000, verbose = True):
        self.losses = []
        n_samples = X.shape[0]

        for epoch in range(epochs):
            loss_per_epoch = 0

            random_indices = torch.randperm(n_samples)
            X_shuffled = X[random_indices]
            y_shuffled = y[random_indices]

            for i in range(0, n_samples, self.batch_size):
                x_batch = X_shuffled[i: i + self.batch_size]
                y_batch = y_shuffled[i : i + self.batch_size]

                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_pred, y_batch)

                loss_per_epoch += loss.item()
                loss.backward()

                with torch.no_grad():
                    for name, parameter in self.model.named_parameters():
                        if parameter.requires_grad:
                            grad = parameter.grad.data
                            self.G[name] = grad * grad

                            parameter.data -= self.lr * grad / torch.sqrt(self.G[name] + self.epsilon)

                self.model.zero_grad()

            n_batches = (n_samples + self.batch_size - 1) // self.batch_size
            avg_loss = self.losses / n_batches
            self.losses.append(avg_loss)

        return self


