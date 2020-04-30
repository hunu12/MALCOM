import abc 

import numpy as np
import torch.nn as nn

class Detector(nn.Module):
    """ A template for OOD detectors to apply alread deployed classifier
    """

    def __init__(self, model, num_classes, ood_tuning=False, **kwargs):
        super(Detector, self).__init__()

        self.model = model
        self.num_classes = num_classes
        self.ood_tuning = ood_tuning

    def forward(self, x):
        out = self.model(x)
        ood_scores = self._get_ood_scores(x)
        return out, ood_scores
    
    @abc.abstractmethod
    def fit(self, train_loader):
        pass

    @abc.abstractmethod
    def tune_parameters(self, valid_loader, out_valid_loader, num_samples):
        pass

    @abc.abstractmethod
    def _get_ood_scores(self):
        pass


class Hook():
    """ This class registers a forward hook for a module.
    The code is from https://www.kaggle.com/sironghuang/understanding-pytorch-hooks
    """
    def __init__(self, module, preprocess=nn.Identity()):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.preprocess = preprocess
    
    def hook_fn(self, module, input_, output):
        #self.input = input_
        self.output = self.preprocess(output)
    
    def close(self):
        self.hook.remove()

def get_scores(detector, dataloader, num_samples=np.inf):
    ood_scores = []
    total = 0
    for data, target in dataloader:
        data = data.cuda()
        total += data.size(0)
        _, scores = detector(data)
        ood_scores.append(scores)
        if total >= num_samples:
            break
    ood_scores = np.concatenate(ood_scores, axis=0)
    if np.isfinite(num_samples):
        ood_scores = ood_scores[:num_samples]
    return ood_scores