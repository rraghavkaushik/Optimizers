# Optimizers

PyTorch Implementation of Optimizers from scratch

## BGD - Batch Gradient Descent
BGD computes the gradient of the cost function w.r.t. to the parameters θ for the entire training dataset. We perform an update in the direction of the gradients and the learning rate(η), determines how large of an update we perform.
Update rule: 

<img width="261" height="56" alt="image" src="https://github.com/user-attachments/assets/6a4a7f54-8ce6-4fb4-8975-a68cc1b2867d" />

Batch gradient descent will converge to the global minimum for convex error surfaces and to a local minimum for non-convex surfaces.


All implementations are based on the paper 'An overview of gradient descent optimization algorithms' by Sebastian Ruder.

## References

```bibtex
@article{ruder2016overview,
  title={An overview of gradient descent optimization algorithms},
  author={Ruder, Sebastian},
  journal={arXiv preprint arXiv:1609.04747},
  year={2016},
  url={https://arxiv.org/abs/1609.04747}
}
```

