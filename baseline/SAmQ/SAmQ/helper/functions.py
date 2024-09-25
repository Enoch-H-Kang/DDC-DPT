# Helper functions

import numpy as np
from datetime import datetime
import os
from dowel import logger, tabular
import dowel


class _Default:  # pylint: disable=too-few-public-methods
    """A wrapper class to represent default arguments.
    Args:
        val (object): Argument value.
    """

    def __init__(self, val):
        self.val = val


def make_optimizer(optimizer_type, module=None, **kwargs):
    """Create an optimizer for pyTorch & tensorflow algos.
    Args:
        optimizer_type (Union[type, tuple[type, dict]]): Type of optimizer.
            This can be an optimizer type such as 'torch.optim.Adam' or a
            tuple of type and dictionary, where dictionary contains arguments
            to initialize the optimizer e.g. (torch.optim.Adam, {'lr' : 1e-3})
        module (optional): If the optimizer type is a `torch.optimizer`.
            The `torch.nn.Module` module whose parameters needs to be optimized
            must be specify.
        kwargs (dict): Other keyword arguments to initialize optimizer. This
            is not used when `optimizer_type` is tuple.
    Returns:
        torch.optim.Optimizer: Constructed optimizer.
    Raises:
        ValueError: Raises value error when `optimizer_type` is tuple, and
            non-default argument is passed in `kwargs`.
    """
    if isinstance(optimizer_type, tuple):
        opt_type, opt_args = optimizer_type
        for name, arg in kwargs.items():
            if not isinstance(arg, _Default):
                raise ValueError('Should not specify {} and explicit \
                    optimizer args at the same time'.format(name))
        if module is not None:
            return opt_type(module.parameters(), **opt_args)
        else:
            return opt_type(**opt_args)

    opt_args = {
        k: v.val if isinstance(v, _Default) else v
        for k, v in kwargs.items()
    }
    if module is not None:
        return optimizer_type(module.parameters(), **opt_args)
    else:
        return optimizer_type(**opt_args)

def makePath(expType, expId = None):
    if expId is None:
        expname = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        expname = expId     
    print('Experiment name: '+expType)
    
    print('Experiment Id: '+expname)
    path = '/home/ubuntu/github/cS/data/'+expType+'/'+expname

    if not os.path.exists(path):
        os.makedirs(path)
    return path, expname


def setupTensorboard(path, message):
    logger.add_output(dowel.StdOutput())
    logger.add_output(dowel.CsvOutput(path+'/progress.csv'))
    logger.add_output(dowel.TextOutput(path+'/progress.txt'))
    logger.add_output(dowel.TensorBoardOutput(path))
    logger.log(message)

def dumpAllTensorboard(names, values):
    for i in range(len(names)):
        tabular.record(names[i], values[i])
    
    

