import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegressionCV

import calculate_log as callog
import cuda_utils as cutils
from detectors import Mahalanobis
from detect_utils import get_scores
import ncd_utils as nutils

class CCP(nn.Module):
    def __init__(self, curves, levels, pmaps):
        self.curves = curves
        self.levels = levels
        self.pmaps = pmaps
        self.func_dict = {}

    def forward(self, x, i):
        out = x.view(x.size(0), x.size(1), -1)
        out = out[:, :, self.curves[i]]
        _, strings = (out.unsqueeze(2) - self.levels[i].unsqueeze(0).unsqueeze(3)).abs().min(2)
        strings = strings.cuda().type(torch.int32).contiguous()

        func = self.func_dict.get((i, strings.size(0)))
        if func is None:
            func = cutils.get_LZW_NCD(strings, self.pmaps[i])
            self.func_dict[(i, strings.size(0))] = func

        ncds = torch.empty((strings.size(0),
                            self.pmaps[i].size(0),
                            self.pmaps[i].size(1))).cuda().type(torch.float32).contiguous()
        func(strings, self.pmaps[i].cuda().type(torch.int32).contiguous(), ncds)
        torch.cuda.synchronize()
        ncds = ncds.view(x.size(0), -1)
        return ncds


class Malcom(Mahalanobis):
    def __init__(self, model, num_classes, ood_tuning=False, **kwargs):
        kwargs['ood_tuning'] = True
        super(Malcom, self).__init__(model, num_classes, **kwargs)
        self.ood_tuning = ood_tuning

    def forward(self, x):
        with torch.no_grad():
            return super().forward(x)

    def fit(self, train_loader, scan='hilbert', num_levels=4):
        # get information about feature extraction
        self.model.eval()
        temp_x = torch.rand([2] + list(train_loader.dataset[0][0].size())).cuda()
        with torch.no_grad():
            self.model(temp_x)

        self.feature_list = []
        for i in range(self.num_features):
            _, c, h, w = self.feature_hooks[i].output.size()
            self.feature_list.append((c, h, w))

        # linearization & quantization
        print("\t...fit transformation")
        self._fit_linearization(scan=scan)
        self._fit_quantization(train_loader, num_levels=num_levels)
        
        # prototypical maps
        print("\t...fit prototypical maps")
        self._fit_prototypical_maps(train_loader)
        self.ccp = CCP(self.curves, self.levels, self.pmaps)

        # Mahalanobis parameters
        print("\t...fit Mahalanobis distance")
        self._fit_Mahalanobis(train_loader)

    def tune_parameters(self, valid_loader, out_valid_loader, num_samples=1000):
        assert self.ood_tuning
        self.regressor = None

        X_in = get_scores(self, valid_loader, num_samples)
        X_out = get_scores(self, out_valid_loader, num_samples)
        X = np.concatenate((X_in, X_out), axis=0)
        Y = np.concatenate((np.zeros(num_samples), np.ones(num_samples)))
        regressor = LogisticRegressionCV(n_jobs=-1).fit(X, Y)
        self.regressor = regressor

    def _get_ood_scores(self, x):
        out_layers = self._get_layers(None, hook=True)
        
        ood_scores = []
        feature_vecs = []
        for i, out_features in enumerate(out_layers):
            out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
            gap = out_features.mean(dim=2)
            ccp = self.ccp.forward(out_features, i)
            twofold_poolings = torch.cat((gap, ccp), dim=1)
            feature_vecs.append(twofold_poolings)

            if self.ood_tuning:
                distance_from_mean = []
                for c in range(self.num_classes):
                    centered = (twofold_poolings - self.class_mean[i][c]).double()
                    md = torch.mm(torch.mm(centered, self.precision[i]), centered.t()).diag()
                    distance_from_mean.append(md)

                scores = torch.stack(distance_from_mean, dim=1).min(dim=1)[0]
                ood_scores.append(scores)
 
        if self.ood_tuning:
            ood_scores = torch.stack(ood_scores, dim=1).cpu().numpy()
            if self.regressor is None:
                return ood_scores

            ood_scores = self.regressor.predict_proba(ood_scores)[:, 1]
            return ood_scores

        feature_vecs = torch.cat(feature_vecs, dim=1)
        distance_from_mean = []
        for c in range(self.num_classes):
            centered = (feature_vecs - self.class_mean[c]).double()
            md = torch.mm(torch.mm(centered, self.precision), centered.t()).diag()
            distance_from_mean.append(md)
        distance_from_mean = torch.stack(distance_from_mean, dim=1)
        ood_scores = distance_from_mean.min(dim=1)[0]
        return ood_scores.cpu().numpy()

    def _fit_linearization(self, scan='hilbert'):
        if scan == 'hilbert':
            self.curves = nutils.hilbert_curves(self.feature_list)
        elif scan == 'vertical':
            self.curves = nutils.vertical_curves(self.feature_list)
        else:
            self.curves = nutils.horizontal_curves(self.feature_list)

    def _fit_quantization(self, train_loader, num_levels=4, tol=1e-1, max_iter=10):
        # warm-up
        data, _ = next(iter(train_loader))
        data = data.cuda()
        out_layers = self._get_layers(data)
        self.levels = [None] * self.num_features
        self.num_levels = num_levels
        for i, (num_maps, _, _) in enumerate(self.feature_list):
            out_features = out_layers[i].permute((1, 0, 2, 3)).reshape(num_maps, -1)
            self.levels[i] = nutils.lloyd_max_quantizer(out_features.double(), num_levels=num_levels, tol=tol, max_iter=500).double()

        # iteration
        old_error = None
        for t in range(max_iter):
            new_levels = [torch.zeros_like(l) for l in self.levels]#torch.zeros_like(levels)
            normalizer = [torch.zeros_like(l) for l in self.levels]
            
            sum_error = 0.0
            for data, _ in train_loader:
                data = data.cuda()
                out_layers = self._get_layers(data)
                for i, (num_maps, H, W) in enumerate(self.feature_list):
                    out_features = out_layers[i].permute((1, 0, 2, 3)).reshape(num_maps, -1).double()
                
                    temp_errors, temp_indices = (out_features.unsqueeze(2) - self.levels[i].unsqueeze(1)).pow(2).min(2)
                    new_levels[i] = new_levels[i].scatter_add_(1, temp_indices, out_features)
                    normalizer[i] = normalizer[i].scatter_add_(1, temp_indices, torch.ones_like(temp_indices, dtype=torch.double))
                    sum_error += (temp_errors / (num_maps * H * W)).sum().item() 

            if old_error is None:
                rel_error = np.inf
                print("\t\tquantization iter 0")
            else:
                rel_error = (old_error - sum_error) / old_error
                print("\t\tquantization iter {}: relative error {:.3f}".format(t, rel_error))
            old_error = sum_error

            if rel_error < 0:
                break
            
            for i in range(len(self.feature_list)):
                normalizer[i][normalizer[i]==0] = 1e-15
                self.levels[i] = new_levels[i]/normalizer[i]
            
            if rel_error < tol:
                break

    def _fit_prototypical_maps(self, train_loader):
        symbol_list = [np.zeros((num_maps, self.num_levels)) for num_maps, _, _ in self.feature_list]
        mean_maps = []
        for num_maps, H, W in self.feature_list:
            temp_maps = torch.zeros((1, num_maps, int(H * W)), dtype=torch.float)
            mean_maps.append(temp_maps.cuda())

        self.model.eval()
        total = 0
        for data, target in train_loader:
            data = data.cuda()
            out_layers = self._get_layers(data)

            for i, out_features in enumerate(out_layers):
                out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
                out_features = out_features[:, :, self.curves[i]]
                
                _, strings = (out_features.unsqueeze(2) - self.levels[i].unsqueeze(0).unsqueeze(3)).abs().min(2)
                strings = strings.type(torch.uint8)

                for l in range(self.num_levels):
                    symbol_list[i][:, l] += (strings==l).sum(dim=(0, 2)).cpu().numpy()
                mean_maps[i] += out_features.float().sum(dim=0, keepdim=True)
                
            total += data.size(0)
    
        self.pmaps = []
        for i in range(self.num_features):
            mean_maps[i] /= total
            symbol_list[i] /= symbol_list[i].sum(axis=1, keepdims=True)
            symbol_list[i] = (symbol_list[i] * mean_maps[i].size(2)).cumsum(axis=1).round().astype(int)[:, :-1]
            symbol_list[i][symbol_list[i]<0] = 0
            symbol_list[i][symbol_list[i]>=mean_maps[i].size(-1)] = mean_maps[i].size(-1)-1

            sorted_values, _ = mean_maps[i].squeeze(0).sort(dim=1)
            temp_threshold = torch.gather(sorted_values, 1, torch.LongTensor(symbol_list[i]).cuda()).unsqueeze(2) ##
            
            strings = torch.zeros_like(mean_maps[i], dtype=torch.uint8)
            for l in range(self.num_levels - 1):
                strings += (mean_maps[i] > temp_threshold[:, l].unsqueeze(0).repeat((mean_maps[i].size(0), 1, 1))).type(torch.uint8)

            self.pmaps.append(strings)

    def _fit_Mahalanobis(self, train_loader):
        self.model.eval()
        if self.ood_tuning:
            feature_vecs = [[list() for _ in range(self.num_classes)] for _ in range(self.num_features)]
        else:
            feature_vecs = [list() for _ in range(self.num_classes)]
        total = 0
        for data, target in train_loader:
            total += data.size(0)
            data = data.cuda()
            out_layers = self._get_layers(data)

            if not self.ood_tuning:
                temp_feature_vecs = []
            for i, out_features in enumerate(out_layers):
                out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
                gap = out_features.mean(dim=2)
                ccp = self.ccp.forward(out_features, i)
                twofold_poolings = torch.cat((gap, ccp), dim=1)

                if self.ood_tuning:
                    for c in range(self.num_classes):
                        feature_vecs[i][c].append(twofold_poolings[torch.nonzero(target==c, as_tuple=False).squeeze(1)].cpu().numpy())
                else:
                    temp_feature_vecs.append(twofold_poolings)

            if not self.ood_tuning:
                temp_feature_vecs = torch.cat(temp_feature_vecs, dim=1) 
                for c in range(self.num_classes):
                    feature_vecs[c].append(temp_feature_vecs[torch.nonzero(target==c, as_tuple=False).squeeze(1)].cpu().numpy())
        
        if self.ood_tuning:
            self.class_mean = [list() for _ in range(self.num_features)]
            self.precision = []
            for i in range(self.num_features):
                X = []
                for c in range(self.num_classes):
                    temp_dists = torch.from_numpy(np.vstack(feature_vecs[i][c]))
                    mean = temp_dists.mean(dim=0, keepdim=True)
                    self.class_mean[i].append(mean.cuda())
                    centered = temp_dists - mean
                    X.append(centered)
                X = torch.cat(X, dim=0).cpu().numpy().astype(np.double)
                tied_cov = (X.T @ X) / X.shape[0]
                precision = np.linalg.pinv(tied_cov)
                self.precision.append(torch.from_numpy(precision).cuda())

        else:
            self.class_mean = []
            X = []
            for c in range(self.num_classes):
                temp_dists = torch.from_numpy(np.vstack(feature_vecs[c]))
                mean = temp_dists.mean(dim=0, keepdim=True)
                self.class_mean.append(mean.cuda())
                centered = (temp_dists - mean).double()
                X.append(centered)
            X = torch.cat(X, dim=0).cpu().numpy().astype(np.double)
            tied_cov = (X.T @ X) / X.shape[0]
            precision = np.linalg.pinv(tied_cov)
            self.precision = torch.from_numpy(precision).cuda()

    def _set_feature_extractor(self, x):
        return super(Malcom, self)._set_feature_extractor(x)
