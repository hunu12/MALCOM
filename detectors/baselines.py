from itertools import product

import numpy as np
import sklearn.covariance
import sklearn.linear_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import calculate_log as callog
from . import detect_utils

class Baseline(detect_utils.Detector):
    """ [ICLR'17] Hendrycks & Gimpel, A baseline for detecting misclassified
    and out-of-distribution examples in neural networks.
    The OOD score is defined as negative max. of softmax.
    """
    def __init__(self, model, num_classes, **kwargs):
        super(Baseline, self).__init__(model, num_classes, **kwargs)

        for _, m in model._modules.items():
            if isinstance(m, nn.Linear):
                self.softmax_layer = detect_utils.Hook(m, nn.Softmax(dim=1))
    
    def forward(self, x):
        with torch.no_grad():
            return super().forward(x)
    
    def fit(self, train_loader):
        pass

    def tune_parameters(self, valid_loader, out_valid_loader, num_samples):
        pass

    def _get_ood_scores(self, x):
        return -self.softmax_layer.output.max(dim=1)[0].cpu().numpy()


class Odin(detect_utils.Detector):
    """ [ICLR'18] Liang et al., Enhancing the reliability of 
    out-of-distribution image de-tection in neural networks.
    The OOD score is defined as negative max. of softmax 
    with the temperature scaling and input perturbation.
    """

    def __init__(self, model, num_classes,
                 normalizer=(1.0, 1.0, 1.0), **kwargs):
        super(Odin, self).__init__(model, num_classes, **kwargs)
        assert self.ood_tuning

        self.criterion = nn.CrossEntropyLoss()
        self.magnitude = None
        self.temperature = None
        self.normalizer = normalizer

        for _, m in model._modules.items():
            if isinstance(m, nn.Linear):
                self.softmax_layer = detect_utils.Hook(m, nn.Softmax(dim=1))
    
    def forward(self, x):
        x = Variable(x, requires_grad=True)
        return super().forward(x)
        
    def fit(self, train_loader):
        pass

    def tune_parameters(self, valid_loader, out_valid_loader,
                        num_samples=1000):
        M_list = [
            0, 0.0005, 0.001, 0.0014, 0.002, 
            0.0024, 0.005, 0.01, 0.05, 0.1, 0.2
        ]
        T_list = [1, 10, 100, 1000]
        best_tnr = -np.inf
        best_m, best_t = None, None
        for magnitude, temperature in product(M_list, T_list):
            self.magnitude, self.temperature = magnitude, temperature

            X_in = detect_utils.get_scores(self, valid_loader, num_samples)
            X_out = detect_utils.get_scores(self, out_valid_loader, num_samples)

            test_results = callog.metric(-X_in, -X_out)
            tnr = test_results['TNR']
            if tnr > best_tnr:
                best_tnr = tnr
                best_m, best_t = magnitude, temperature
        self.magnitude, self.temperature = best_m, best_t

    def _get_ood_scores(self, x):
        assert self.magnitude is not None
        assert self.temperature is not None

        output = self.softmax_layer.output
        scaled_logit = output / self.temperature

        labels = Variable(scaled_logit.data.max(1)[1])
        
        loss = self.criterion(scaled_logit, labels)
        loss.backward()

        # input perturbation
        gradient = x.grad.data
        with torch.no_grad():
            indices = (gradient >= 0)
            gradient[indices] = 1.0
            gradient[~indices] = -1.0
            for i in range(3):
                gradient[:, i, :, :] /= self.normalizer[i]

            noise_x = x - self.magnitude * gradient
            noise_output = self.model(noise_x)
            noise_scaled_logit = noise_output / self.temperature
            score = - F.softmax(noise_scaled_logit, dim=1).max(dim=1)[0]
        return score.cpu().numpy()
        

class Mahalanobis(detect_utils.Detector):
    """ [NeurIPS'18] Lee et al., A simple unified framework for detecting 
    out-of-distribution samples and adversarial attacks.
    The OOD score is defined as min. of Mahalanobis distances from class means
    with the logistic regression and input perturbation.
    """

    def __init__(self, model, num_classes, net_type='', 
                 normalizer=(1.0, 1.0, 1.0), **kwargs):
        super(Mahalanobis, self).__init__(model, num_classes, **kwargs)
        assert self.ood_tuning

        self.magnitude = None
        self.regressor = None
        self.normalizer = normalizer
        self.feature_hooks = []

        for m_name, m in model._modules.items():
            if net_type == 'densenet':
                if m_name in ['conv1', 'trans1', 'trans2', 'relu']:
                    self.feature_hooks.append(detect_utils.Hook(m))
            elif net_type == 'resnet':
                if m_name == 'bn1':
                    self.feature_hooks.append(
                        detect_utils.Hook(m, nn.ReLU(inplace=False))
                    )
                elif m_name in ['layer1', 'layer2', 'layer3', 'layer4',]:
                    self.feature_hooks.append(detect_utils.Hook(m))
            else:
                self._set_feature_extractor(m)

        if len(self.feature_hooks) > 5:
            indices = np.linspace(0, len(self.feature_hooks)-1, 5, 
                                  endpoint=True).astype(int)
            self.feature_hooks = [self.feature_hooks[i] for i in indices]
        self.num_features = len(self.feature_hooks)

    def forward(self, x):
        return super().forward(x)
        
    def fit(self, train_loader):
        self.model.eval()
        
        # Mahalanobis parameters
        print("\t...fit Mahalanobis distance")
        self._fit_Mahalanobis(train_loader)
        return

    def tune_parameters(self, valid_loader, out_valid_loader, num_samples=1000):
        self.regressor = None
        num_train = num_samples // 2

        M_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]

        best_tnr = -np.inf
        best_m, best_regressor = None, None
        for magnitude in M_list:
            self.magnitude = magnitude

            X_in = detect_utils.get_scores(self, valid_loader, num_samples)
            X_out = detect_utils.get_scores(self, out_valid_loader, 
                                            num_samples)

            X_train = np.concatenate((X_in[:num_train], X_out[:num_train]), 
                                    axis=0)
            Y_train = np.concatenate((np.zeros(num_train), np.ones(num_train)))
            regressor = sklearn.linear_model.LogisticRegressionCV(n_jobs=-1)\
                        .fit(X_train, Y_train)
            
            X_valid_in = regressor.predict_proba(
                X_in[num_train:num_samples]
            )[:, 1]
            X_valid_out = regressor.predict_proba(
                X_out[num_train:num_samples]
            )[:, 1]

            test_results = callog.metric(-X_valid_in, -X_valid_out)
            tnr = test_results['TNR']
            if tnr > best_tnr:
                best_tnr = tnr
                best_m, best_regressor = magnitude, regressor
        self.magnitude, self.regressor = best_m, best_regressor

    def _get_ood_scores(self, x):
        assert self.magnitude is not None

        ood_scores = []
        for i in range(self.num_features):
            _x = Variable(x, requires_grad=True)
            out_features = self._get_intermediate_layer(_x, i)
            out_features = out_features.view(
                out_features.size(0), out_features.size(1), -1
            )
            gap = out_features.mean(dim=2)

            distance_from_mean = []
            for c in range(self.num_classes):
                centered = (gap - Variable(self.class_mean[i][c])).double()
                md = torch.mm(
                    torch.mm(centered, Variable(self.precision[i])), 
                    centered.t()
                ).diag()
                distance_from_mean.append(md)
            pure_scores = torch.stack(distance_from_mean, dim=1).min(dim=1)[0]

            # input preprocessing
            pure_scores.mean().backward()
            gradient = _x.grad.data
            with torch.no_grad():
                indices = (gradient >= 0)
                gradient[indices] = 1.0
                gradient[~indices] = -1.0
                for i in range(3):
                    gradient[:, i, :, :] /= self.normalizer[i]
                
                noise_x = _x + self.magnitude * gradient
                out_features = self._get_intermediate_layer(noise_x, i)
                out_features = out_features.view(
                    out_features.size(0), out_features.size(1), -1
                )
                gap = out_features.mean(dim=2)

                distance_from_mean = []
                for c in range(self.num_classes):
                    centered = (gap - Variable(self.class_mean[i][c])).double()
                    md = torch.mm(
                        torch.mm(centered, Variable(self.precision[i])),
                        centered.t()
                    ).diag()
                    distance_from_mean.append(md)
                scores = torch.stack(distance_from_mean, dim=1).min(dim=1)[0]
                ood_scores.append(scores)
 
        ood_scores = torch.stack(ood_scores, dim=1).cpu().numpy()
        if self.regressor is None:
            return ood_scores

        ood_scores = self.regressor.predict_proba(ood_scores)[:, 1]
        return ood_scores

    def _fit_Mahalanobis(self, train_loader):
        self.model.eval()
        feature_vecs = [[list() for _ in range(self.num_classes)] for _ in range(self.num_features)]
        total = 0
        for data, target in train_loader:
            total += data.size(0)
            data = data.cuda()
            with torch.no_grad():
                out_layers = self._get_layers(data)

            for i, out_features in enumerate(out_layers):
                out_features = out_features.view(
                    out_features.size(0), out_features.size(1), -1
                )
                gap = out_features.mean(dim=2)
                for c in range(self.num_classes):
                    _indices = torch.nonzero(
                        target==c, as_tuple=False
                    ).squeeze(1)
                    feature_vecs[i][c].append(gap[_indices].cpu().numpy())

        self.class_mean = [list() for _ in range(self.num_features)]
        self.precision = []
        for i in range(self.num_features):
            X = []
            for c in range(self.num_classes):
                temp_dists = torch.from_numpy(np.vstack(feature_vecs[i][c]))
                mean = temp_dists.mean(dim=0, keepdim=True)
                self.class_mean[i].append(mean.cuda())
                centered = (temp_dists - mean).double()
                X.append(centered)
            X = torch.cat(X, dim=0).cpu().numpy().astype(np.double)

            tied_cov = sklearn.covariance.EmpiricalCovariance(
                store_precision=True, assume_centered=False
            )
            tied_cov.fit(X)
            precision = tied_cov.get_precision()
            self.precision.append(torch.from_numpy(precision).cuda())


    def _set_feature_extractor(self, x):
        if isinstance(x, nn.Conv2d):
            self.feature_hooks.append(detect_utils.Hook(x))
        
        for c in x.children():
            self._set_feature_extractor(c)

    def _get_feature_information(self, train_loader):
        # get information about feature extraction
        temp_x = torch.rand(
            [2] + list(train_loader.dataset[0][0].size())
        ).cuda()
        temp_x = Variable(temp_x)
        with torch.no_grad():
            self.model(temp_x)

        self.feature_list = []
        for i in range(self.num_features):
            _, c, h, w = self.feature_hooks[i].output.size()
            self.feature_list.append((c, h, w))
    
    def _get_layers(self, x, hook=False):
        if not hook:
            with torch.no_grad():
                self.model(x)
        return [self.feature_hooks[i].output for i in range(self.num_features)]

    def _get_intermediate_layer(self, x, i, hook=False):
        if not hook:
            self.model(x)
        return self.feature_hooks[i].output

    

        