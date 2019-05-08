import numpy as np
from collections import namedtuple

import torch
from torch.nn import functional as F
from torch.distributions.uniform import Uniform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import Transform, AffineTransform, PowerTransform
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

# might be importable from torch.distributions.utils in a future pytorch version
euler_constant = 0.57721566490153286060


class Kumaraswamy(TransformedDistribution):
    r"""
    Samples from a Kumaraswamy distribution.

    Example:

        >>> m = Kumaraswamy(torch.Tensor([1.0]), torch.Tensor([1.0]))
        >>> m.sample()  # sample from a Kumaraswamy distribution with concentration alpha=1 and beta=1
         0.1729
        [torch.FloatTensor of size (1,)]

    Args:
        a (float or Tensor): 1st concentration parameter of the distribution
            (often referred to as alpha)
        b (float or Tensor): 2nd concentration parameter of the distribution
            (often referred to as beta)
    """
    arg_constraints = {'a': constraints.positive, 'b': constraints.positive}
    support = constraints.unit_interval
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        finfo = torch.finfo(self.b.dtype)

        # for stability Uniform(0, 1) is defined as Uniform(eps, 1-eps)
        base_dist = Uniform(torch.full_like(self.b, finfo.eps),
                            torch.full_like(self.b, 1 - finfo.eps))
        transforms = [AffineTransform(loc=1, scale=-torch.ones_like(self.a)),
                      PowerTransform(exponent=self.b.reciprocal()),
                      AffineTransform(loc=1, scale=-torch.ones_like(self.b)),
                      PowerTransform(exponent=self.a.reciprocal())]
        super(Kumaraswamy, self).__init__(base_dist, transforms, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Kumaraswamy, _instance)
        new.a = self.a.expand(batch_shape)
        new.b = self.b.expand(batch_shape)
        return super(Kumaraswamy, self).expand(batch_shape, _instance=new)

    @property
    def mean(self):
        return _moments(self.a, self.b, 1)

    @property
    def variance(self):
        return _moments(self.a, self.b, 2) - torch.pow(self.mean, 2)

    def entropy(self):
        t1 = (1 - self.a.reciprocal())
        t0 = (1 - self.b.reciprocal())
        H0 = torch.digamma(self.b + 1) + euler_constant
        return t1 + t0 * H0 + torch.log(self.a) + torch.log(self.b)

def _moments(a, b, n):
    """
    Computes nth moment of Kumaraswamy using using torch.lgamma
    """
    arg1 = 1 + n / a
    log_value = torch.lgamma(arg1) + torch.lgamma(b) - torch.lgamma(arg1 + b)
    return b * torch.exp(log_value)


class StretchedKumaraswamy(TransformedDistribution):
    r"""
    Creates a Stretched Kumaraswamy distribution parametrized by non-negative shape parameters
    :attr:`a`, and :attr:`b` and with support."""

    arg_constraints = {'a': constraints.positive, 'b': constraints.positive}
    support = constraints.simplex
    has_rsample = True

    def __init__(self, a, b, support, validate_args=None):

        loc = support[0]
        scale = support[1] - support[0]

        super(StretchedKumaraswamy, self).__init__(
            Kumaraswamy(a, b), AffineTransform(loc=loc, scale=scale),
            validate_args=validate_args)


class HardKumaraswamy(StretchedKumaraswamy):

    r"""
    Creates a Kumaraswamy distribution parametrized by positive shape parameters
    :attr:`a`, and :attr:`b`.

    Example::

        >>> m = Kumaraswamy(torch.tensor([1.]), torch.tensor([1.]))
        >>> m.sample()
        tensor([0.5370])

    Args:
        a (Tensor): positive shape parameter
        b (Tensor): positive shape parameter
    """
    arg_constraints = {'a': constraints.positive, 'b': constraints.positive}
    support = constraints.integer_interval
    has_rsample = True

    def __init__(self, a, b, support, validate_args=None):
        super(HardKumaraswamy, self).__init__(a, b, support, validate_args=validate_args)

    def sample(self):
        return F.hardtanh(super().sample(), min_val=0., max_val=1.)

    def rsample(self):
        return F.hardtanh(super().rsample(), min_val=0., max_val=1.)
