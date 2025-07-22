''' Implementation of AdaMax. 
AdaMax is very similar to Adam but differs slightly in the second moment (variance) updates. (Refering to my Adam's
implementation before coming to this would be more helpful.)
Adam scales the gradient inversely proportional to the l2 norm of the past gradients and the current gradient.
We can generalize this update to the lp norm. Usually the l1 and l2 norms are numerically stable as the p value
increase, the norms become unstable. But l_inf is also found to be stable. So the authors of AdaMax, propose that 
using l_inf leads to a more stable value. 

(For any vector x, ||x||∞ = max(|x₁|, |x₂|, ..., |xₙ|)
here, l_inf - l∞ 
)

Update rules:
m_t = β1 * m_t−1 + (1 − β1) * g_t
u_t = max(β2 * u_t−1, |g_t|)

m_t - the estimate of the first moment (the mean)
u_t - the estimate of the second moment (uncentred variance)

m^_t = m_t / (1 − β1 ** t)
bias-corrected moment estimate - m^_t

θ_t+1 = θ_t − η * m^_t / u_t

Note: u_t relies on the max operation, it is not as suggestible to bias towards zero as m_t and v_t
in Adam, which is why we do not need to compute a bias correction for u_t.


Look at Sectrion 4.7 of the paper 'An overview of gradient descent optimization
algorithms' by Sebastian Ruder for the math behind this.

'''

import torch
import torch.nn as nn

class AdaMax:
    def __init__(self, model, loss_fn = nn.MSELoss(), lr = 0.002, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, batch_size = 32, ):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        # self.eps = eps
        self.batch_size = batch_size
        self.losses = []
        self.step_count = 0

        self.m = {}
        self.u = {}

        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.m[name] = torch.zeros_like(parameter.data)
                self.u[name] = torch.zeros_like(parameter.data)

    def fit(self, X, y, epochs=1000, verbose=True):
        self.losses = []
        self.step_count = 0

        n_samples = X.shape[0]

        for epoch in range(epochs):
            loss_per_epoch = 0
            random_indices = torch.randperm(n_samples)

            X_shuffled = X[random_indices]
            y_shuffled = y[random_indices]

            for i in range(0, n_samples, self.batch_size):
                if i + self.batch_size > n_samples:
                    x_batch = X_shuffled[i:n_samples]
                    y_batch = y_shuffled[i:n_samples]
                else:
                    x_batch = X_shuffled[i:i + self.batch_size]
                    y_batch = y_shuffled[i:i + self.batch_size]

                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_pred, y_batch)
                loss_per_epoch += loss.item()

                loss.backward()
                self.step_count += 1

                with torch.no_grad():
                    for name, parameter in self.model.named_parameters():
                        if parameter.requires_grad and parameter.grad is not None:

                            grad = parameter.grad.data

                            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
                            self.u[name] = torch.max(self.beta2 * self.u[name], torch.abs(grad))

                            m_hat = self.m[name] / (1 - self.beta1 ** self.step_count)

                            parameter.grad.data -= self.lr * m_hat / self.u[name]

                self.model.zero_grad()

            n_batches = (n_samples + self.batch_size - 1) // self.batch_size
            avg_loss = loss_per_epoch / n_batches
            self.losses.append(avg_loss)

            if verbose and epoch % 50 == 0:
                print(f"epoch {epoch}, loss: {avg_loss}")
        
        return self



