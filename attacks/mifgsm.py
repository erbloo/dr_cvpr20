import copy
import numpy as np
from torch.autograd import Variable
import torch

import pdb

class MomentumIteratorAttack(object):
    def __init__(self, model, decay_factor=1, epsilon=0.3, steps=40, step_size=0.01, 
        random_start=False):
        """
        The Momentum Iterative Fast Gradient Sign Method (Dong et al. 2017).
        This method won the first places in NIPS 2017 Non-targeted Adversarial
        Attacks and Targeted Adversarial Attacks. The original paper used
        hard labels for this attack; no label smoothing. inf norm.
        Paper link: https://arxiv.org/pdf/1710.06081.pdf
        """

        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.rand = random_start
        self.model = copy.deepcopy(model)
        self.loss_fn = torch.nn.CrossEntropyLoss().cuda()
        self.decay_factor = decay_factor

    def __call__(self, X_nat, y):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        X_nat_np = X_nat.numpy()
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.model.eval()
        if self.rand:
            X = X_nat_np + np.random.uniform(-self.epsilon, self.epsilon,
                X_nat_np.shape).astype('float32')
        else:
            X = np.copy(X_nat_np)
        
        momentum = 0
        for _ in range(self.steps):
            X_var = Variable(torch.from_numpy(X).cuda(), requires_grad=True, volatile=False)
            y_var = y.cuda()
            scores = self.model(X_var)
            
            loss = self.loss_fn(scores, y_var)
            self.model.zero_grad()
            loss.backward()
            grad = X_var.grad.data.cpu().numpy()
            X_var.grad.zero_()
            velocity = grad / np.sum(np.absolute(grad))
            momentum = self.decay_factor * momentum + velocity

            X += self.step_size * np.sign(momentum)
            X = np.clip(X, X_nat_np - self.epsilon, X_nat_np + self.epsilon)
            X = np.clip(X, 0, 1)
        return torch.from_numpy(X)