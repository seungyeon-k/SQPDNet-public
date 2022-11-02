from torch.nn import Module

"""Performs :math:`L_p` normalization of inputs over specified dimension.
    Does:
    
    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}
    
    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\lVert v \rVert_p` is not a matrix
    norm.
    
    With default arguments normalizes over the second dimension with Euclidean
    norm.
    
    Args:
        input: input tensor of any shape
        p (float): the exponent value in the norm formulation
        dim (int): the dimension to reduce
        eps (float): small value to avoid division by zero
"""

## FWD/BWD pass module
class Normalize(Module):
    def __init__(self, p=2, dim=1, eps=1e-12):
        super(Normalize, self).__init__()
        self.p, self.dim, self.eps = p, dim, eps

    def forward(self, input):
        return input / input.norm(self.p, self.dim).clamp(min=self.eps).expand_as(input)