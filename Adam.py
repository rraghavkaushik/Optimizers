''' Implementation of Adaptive Moment Estimation (Adam) Optimizer.
Adam is a method that computes adaptive learning rates for each parameter by keeping 
track of exponentially decaying average of past squared gradients (like Adadelta & RMSprop) 
and also keeps track of exponentially decaying average of past gradients (like momentum).
(An overview of gradient descent optimization algorithms (2017), S. Ruder.)

Update Rules:
m_t = β1 * m_t−1 + (1 − β1) * g_t
v_t = β2 * v_t−1 + (1 − β2) * (g_t ** 2)

m_t - the estimate of the first moment (the mean)
v_t - the estimate of the second moment (uncentred variance)

To conteract biases towards 0 in the initial time steps becuase of the decay rates (β1 & β2 are small, close to 1), we 
compute bias-corrected moment estimates, m^_t and v^_t,

m^_t = m_t / (1 − β1 ** t)
v^_t = v_t / (1 - β2 ** t)

Parameter Update: 

θ_t+1 = θ_t − η * m^_t / √(v^_t) + ϵ

θ - parameter
η - learning rate (the Adam paper uses alpha, so I have used that for my implementation)
m^_t - bias-corrected first moment estimate
v^_t - bias-corrected second moment estimate
'''

import torch
import torch.nn as nn

class Adam:
    def __init__(self, model, loss_fn = nn.MSELoss(), beta1 = 0.9, beta2 = 0.999, alpha = 0.001, eps = 1e-8, batch_size = 32):
        self.model = model
        self.loss_fn = loss_fn
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        self.eps = eps
        self.batch_size = batch_size
        self.step_count = 0
        self.losses = []

        self.m_t = {}
        self.v_t = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.m_t[name] = torch.zeros_like(parameter.data)
                self.v_t[name] = torch.zeros_like(parameter.data)

    def fit(self, X, y, epochs = 1000, verbose = True):
        ''' The losses and the step_count values are re_initialised to resuse this function or to train this optimizer 
        again. '''

        self.losses = []
        self.step_count = 0
        n_samples = X.shape[0]

        for epoch in epochs:
            loss_per_epoch = 0

            random_indices = torch.randperm(n_samples)
            X_shuffled = X[random_indices]
            Y_shuffled = y[random_indices]

            for i in range(0 , n_samples, self.batch_size):
                x_batch = X_shuffled[i: i + self.batch_size]
                y_batch = Y_shuffled[i: i + self.batch_size]

                y_pred = self.model(x_batch)

                loss = self.loss_fn(y_pred, y_batch)
                self.step_count += 1

                loss_per_epoch += loss.item()
                loss.backward()

                with torch.no_grad():
                    for name, parameter in self.model.named_parameters():
                        if parameter.requires_grad and parameter is not None:
                            g_t = parameter.grad.data

                            self.m_t[name] = self.beta1 * self.m_t[name] + (1 - self.beta1) * g_t
                            self.v_t[name] = self.beta2 * self.v_t[name] + (1 - self.beta2) * g_t * g_t

                            m_hat = self.m_t[name] / (1 - self.beta1 ** self.step_count)
                            v_hat = self.v_t[name] / (1 - self.beta2 ** self.step_count)

                            parameter.grad.data -= self.alpha * m_hat / (torch.sqrt(v_hat) + self.eps)

                self.model.zero_grad()

            n_batches = (n_samples + self.batch_size - 1) // self.batch_size
            avg_loss = loss_per_epoch / n_batches
            self.losses.append(avg_loss)

            if verbose and epoch % 50 == 0:
                print(f'epoch {epoch}, loss: {avg_loss:.4f}')
                
        return self
