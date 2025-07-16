import torch
import torch.nn as nn

class Adadelta:
    def __init__(self, model, loss_fn = nn.MSELoss(), lr = 0.01, momentum = 0.9, eps = 1e-8, batch_size = 32):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.momentum = momentum
        self.eps = eps
        self.batch_size = batch_size
        self.losses = []
        self.E_g2 = {}
        self.theta_2 = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.E_g2[name] = torch.zeros_like(parameter.data)
                self.theta_2[name] = torch.zeros_like(parameter.data)

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
                    for name, parameter in self.model.named_paramters():
                        if parameter.requires_grad and parameter is not None:
                            self.E_g2[name] = self.momentum * self.E_g2[name] + (1 - self.momentum) * parameter.grad.data * parameter.grad.data
                            rms_g = torch.sqrt(self.E_g2[name] + self.eps)
                            rms_theta = torch.qrt(self.theta_2[name] + self.eps)

                            delta_theta = -(rms_theta * parameter.grad.data) / rms_g
                            parameter.grad.data += delta_theta

                            self.theta_2[name] = self.momentum * self.theta_2[name] + (1 - self.momentum) * delta_theta * delta_theta
                self.model.zero_grad()

            n_batches = (self.batch_size + n_samples - 1) // self.batch_size
            avg_loss = loss_per_epoch / n_batches
            self.losses.append(avg_loss)

            if verbose and epoch % 50 == 0:
                print(f'epoch {epoch}, loss: {loss}')

        return self
