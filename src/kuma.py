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


def _lbeta(a, b):
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)


def _harmonic_number(x):
    return torch.digamma(x + 1) + euler_constant


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
            return (1 - self.b_reciprocal) + (1 - self.a_reciprocal) * _harmonic_number(self.b) \
                    - torch.log(self.a) - torch.log(self.b)

    @register_kl(Kumaraswamy, Kumaraswamy)
    def _kl_kumaraswamy_kumaraswamy(p, q):
        x = p.sample(sample_shape=torch.Size([1]))  # TODO: more samples?
        p_entropy = p.entropy()  # TODO: exact or estimate p.log_prob(x).mean(0)?
        cross_entropy = - q.log_prob(x).mean(0)
        return - p_entropy + cross_entropy

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

    def log_prob(self, value):
        # x ~ p(x), and y = hardtanh(x) -> y ~ q(y)

        # log_q(x==0) = cdf_p(0) and log_q(x) = log_q(x)
        zeros = torch.zeros_like(value)
        log_p = torch.where(value == zeros,
                            torch.log(self.stretched.cdf(zeros)),
                            self.stretched.log_prob(value))

        # log_q(x==1) = 1 - cdf_p(1)
        ones = torch.ones_like(value)
        log_p = torch.where(value == ones,
                            torch.log(1 - self.stretched.cdf(ones)),
                            log_p)

        return log_p

    def cdf(self, value):
        """
        Note that HardKuma.cdf(0) = HardKuma.pdf(0) by definition of HardKuma.pdf(0),
         also note that HardKuma.cdf(1) = 1 by definition because
         the support of HardKuma is the *closed* interval [0, 1]
         and not the open interval (left, right) which is the support of the stretched variable.
        """
        cdf = torch.where(
            value < torch.ones_like(value),
            self.stretched.cdf(value),
            torch.ones_like(value)  # all of the mass
        )
        return cdf

    @register_kl(HardKumaraswamy, HardKumaraswamy)
    def _kl_hardkumaraswamy_hardkumaraswamy(p, q):
        assert type(p.base) is type(q.base) and p.loc == q.loc and p.scale == q.scale

        # wrt (lower, upper)
        x = p.sample(sample_shape=torch.Size([10]))  # TODO: more samples?
        estimate = (p.log_prob(x) - q.log_prob(x)).mean(0)

        return estimate

    #@register_kl(HardKumaraswamy, HardKumaraswamy)
    def __kl_hardkumaraswamy_hardkumaraswamy(p, q):
        assert type(p.base) is type(q.base) and p.loc == q.loc and p.scale == q.scale

        # wrt (lower, upper)
        proposal = p.base
        x = proposal.sample(sample_shape=torch.Size([1]))  # TODO: more samples?
        klc = (torch.exp(p.log_prob(x) - proposal.log_prob(x)) * (p.log_prob(x) - q.log_prob(x))).mean(0)

        # wrt lower
        zeros = torch.zeros_like(klc)
        log_p0 = p.log_prob(zeros)
        p0 = torch.exp(log_p0)
        log_q0 = q.log_prob(zeros)
        kl0 = p0 * (log_p0 - log_q0)

        # wrt upper
        ones = torch.ones_like(klc)
        log_p1 = p.log_prob(ones)
        p1 = torch.exp(log_p1)
        log_q1 = q.log_prob(ones)
        kl1 = p1 * (log_p1 - log_q1)

        return kl0 + kl1 + klc
