import torch.nn as nn
import torch

"""Implements the EmbeddingMul class
Author: Noémien Kocher
Date: Fall 2018
Unit test: embedding_mul_test.py
"""

class EmbeddingMul(nn.Module):
    """This class implements a custom embedding mudule which uses matrix
    multiplication instead of a lookup. The method works in the functional
    way.
    Note: this class accepts the arguments from the original pytorch module
    but only with values that have no effects, i.e set to False, None or -1.
    """

    def __init__(self, depth, device):
        super(EmbeddingMul, self).__init__()
        # i.e the dictionnary size
        self.depth = depth
        self.device = device
        self.ones = torch.eye(depth, requires_grad=False)#, device=self.device)
        self._requires_grad = False
        # "oh" means One Hot
        self.last_oh = None
        self.last_weight = None

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self._requires_grad = value
        logger.info(
            f"(embedding mul) requires_grad set to {self.requires_grad}. ")

    def forward(self, input, weight, padding_idx=None, max_norm=None,
                norm_type=2., scale_grad_by_freq=False, sparse=False):
        """Declares the same arguments as the original pytorch implementation
        but only for backward compatibility. Their values must be set to have
        no effects.
        Args:
            - input: of shape (bptt, bsize)
            - weight: of shape (dict_size, emsize)
        Returns:
            - result: of shape (bptt, bsize, dict_size)
        """
        # ____________________________________________________________________
        # Checks if unsupported argument are used
        if padding_idx != -1:
            raise NotImplementedError(
                f"padding_idx must be -1, not {padding_idx}")
        if max_norm is not None:
            raise NotImplementedError(f"max_norm must be None, not {max_norm}")
        if scale_grad_by_freq:
            raise NotImplementedError(f"scale_grad_by_freq must be False, "
                                      f"not {scale_grad_by_freq}")
        if sparse:
            raise NotImplementedError(f"sparse must be False, not {sparse}")
        # ____________________________________________________________________

        if self.last_oh is not None:
            del self.last_oh
        self.last_oh = self.to_one_hot(input)

        with torch.set_grad_enabled(self.requires_grad):
            result = torch.stack(
                [torch.mm(batch.float(), weight)
                 for batch in self.last_oh], dim=0)
        self.last_weight = weight.clone()
        return result

    def to_one_hot(self, input):
        # Returns a new tensor that doesn't share memory
        result = torch.index_select(
            self.ones, 0, input.view(-1).long()).view(
            input.size()+(self.depth,))
        result.requires_grad = self.requires_grad
        return result

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


if __name__ == "__main__":
    input = torch.tensor([[1, 2, 0], [3, 4, 5]])
    dim = 10
    mod = EmbeddingMul(dim)
    emmatrix = torch.rand(10, 5)
    print(emmatrix)
    output = mod(input, emmatrix, -1)
    print(output)
