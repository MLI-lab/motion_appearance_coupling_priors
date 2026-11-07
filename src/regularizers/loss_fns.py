import torch

class MSELoss(torch.nn.Module):
    """
    Mean Squared Error loss function
    """
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, input, target, **kwargs):
        return self.mse(input, target)


def loss_fn_resolver(name, **params):
    if name == "mse":
        return MSELoss()