from typing import Tuple

import torch
import torch.nn as nn

from torch import Tensor


class PassThrough(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, score) -> Tensor:
        return score(input1, input2)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None]:
        return grad_output[:,:3], None, None
    
class ScoreWithIdentityGradWrapper(nn.Module):
    def __init__(self, score):
        super().__init__()
        self.score = score
        self.fn = PassThrough.apply

    def forward(self, input1, input2):
        return self.fn(input1, input2, self.score)



