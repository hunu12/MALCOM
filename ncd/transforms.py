import numpy as np
import torch

def sgn(x):
    return (x > 0) - (x < 0)

def gilbert2d(x, y, ax, ay, bx, by):
    """Generalized Hilbert ('gilbert') space-filling curve for arbitrary-sized
    2D rectangular grids.
    original code is from https://github.com/jakubcerveny/gilbert
    """
    scan = []
    
    w = abs(ax + ay)
    h = abs(bx + by)

    (dax, day) = (sgn(ax), sgn(ay)) # unit major direction
    (dbx, dby) = (sgn(bx), sgn(by)) # unit orthogonal direction

    if h == 1:
        # trivial row fill
        for i in range(0, w):
            scan.append((x, y))
            (x, y) = (x + dax, y + day)
        return scan

    if w == 1:
        # trivial column fill
        for i in range(0, h):
            scan.append((x, y))
            (x, y) = (x + dbx, y + dby)
        return scan

    (ax2, ay2) = (ax//2, ay//2)
    (bx2, by2) = (bx//2, by//2)

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2*w > 3*h:
        if (w2 % 2) and (w > 2):
            # prefer even steps
            (ax2, ay2) = (ax2 + dax, ay2 + day)

        # long case: split in two parts only
        scan += gilbert2d(x, y, ax2, ay2, bx, by)
        scan += gilbert2d(x+ax2, y+ay2, ax-ax2, ay-ay2, bx, by)

    else:
        if (h2 % 2) and (h > 2):
            # prefer even steps
            (bx2, by2) = (bx2 + dbx, by2 + dby)

        # standard case: one step up, one long horizontal, one step down
        scan += gilbert2d(x, y, bx2, by2, ax2, ay2)
        scan += gilbert2d(x+bx2, y+by2, ax, ay, bx-bx2, by-by2)
        scan += gilbert2d(x+(ax-dax)+(bx2-dbx), y+(ay-day)+(by2-dby),
                 -bx2, -by2, -(ax-ax2), -(ay-ay2))
    return scan

def hilbert_curves(feature_list):
    curves = []
    for _, H, W in feature_list:
        if W < H:
            temp_curve = gilbert2d(0, 0, H, 0, 0, W)
        else:
            temp_curve = gilbert2d(0, 0, 0, W, H, 0)

        indices = []
        for i, j in temp_curve:
            indices.append(i * W + j)
        curves.append(indices)
    return curves

def vertical_curves(feature_list):
    curves = []
    for _, H, W in feature_list:
        indices = []
        for ii in range(int(H * W)):
            y = ii % H
            x = ii // H
            indices.append(int(y * W + x))
        curves.append(indices)
    return curves

def horizontal_curves(feature_list):
    curves = []
    for _, H, W in feature_list:
        indices = list(range(int(H * W)))
        curves.append(indices)
    return curves

def lloyd_max_quantizer(x, num_levels=4, max_iter=500, tol=1e-5):
    # initialization
    minimums = x.min(dim=1)[0].unsqueeze(1)
    scale = x.max(dim=1)[0].unsqueeze(1) - minimums
    reps = torch.cat([minimums + (i+1/2) / num_levels * scale 
                      for i in range(num_levels)], dim=1)

    old_error = None
    for _ in range(max_iter):
        temp_errors, temp_indices = (
            x.unsqueeze(2) - reps.unsqueeze(1)
        ).pow(2).min(2)

        new_reps = torch.zeros_like(reps).scatter_add_(1, temp_indices, x)
        normalizer = torch.zeros_like(reps).scatter_add_(
            1, temp_indices, torch.ones_like(temp_indices, dtype=x.dtype)
        )
        normalizer[normalizer==0] += 1e-15

        reps = new_reps / normalizer
        error = temp_errors.sum().item()
        if old_error is None:
            rel_error = np.inf
        else:
            rel_error = (old_error - error) / old_error
        old_error = error

        if rel_error < tol:
            return reps

    print("Warning : quantizer does not converge")
    return reps