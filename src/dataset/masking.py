# Based on matlab code provided by the CMRxRecon challenge:
#  https://github.com/CmrxRecon/CMRxRecon2024/tree/main/CMRxReconMaskGeneration/Task2/Toolbox_Mask_Generator_Task2
#  If you want to use the code, please cite the following paper:
#  Zi Wang et al., Deep Separable Spatiotemporal Learning for Fast Dynamic Cardiac MRI, arXiv:2402.15939, 2024.

import numpy as np

from math import ceil, floor
from scipy.linalg import toeplitz
from scipy.ndimage import rotate

def zpad(x, sx, sy, sz=None, st=None):
    s = [sx, sy]
    if sz is not None:
        s.append(sz)
    if st is not None:
        s.append(st)
    
    m = list(x.shape)
    if len(m) < len(s):
        m.extend([1] * (len(s) - len(m)))
    
    if m == s:
        return x
    
    res = np.zeros(s, dtype=x.dtype)
    idx = [slice(floor(s[i]/2) + ceil(-m[i]/2), floor(s[i]/2) + ceil(m[i]/2)) for i in range(len(s))]
    res[tuple(idx)] = x
    return res

def imrotate(image, angle, mode='constant'):
    return rotate(image, angle, reshape=False, mode=mode)

def crop(x, sx, sy):
    cx, cy = x.shape[0] // 2, x.shape[1] // 2
    return x[cx - sx // 2:cx + sx // 2, cy - sy // 2:cy + sy // 2]

def ktdup(ph, ti, ny, nt):
    ph = ph + ceil((ny + 1) / 2)
    ti = ti + ceil((nt + 1) / 2)
    pt = (ti - 1) * ny + ph

    uniquept, countOfpt = np.unique(pt, return_counts=True)
    repeatedValues = uniquept[countOfpt != 1]

    if len(repeatedValues) == 0:
        # No duplicates, return original values
        ph = ph - ceil((ny + 1) / 2)
        ti = ti - ceil((nt + 1) / 2)
        return ph, ti
    
    dupind = np.concatenate([np.where(pt == val)[0][1:] for val in repeatedValues])
    empind = np.setdiff1d(np.arange(1, ny * nt + 1), pt)

    for i in dupind:
        newind = nearestvac(pt[i], empind, ny)
        pt[i] = newind
        empind = np.setdiff1d(empind, newind)

    ph, ti = ind2xy(pt, ny)
    ph = ph - ceil((ny + 1) / 2)
    ti = ti - ceil((nt + 1) / 2)
    return ph, ti

def nearestvac(dupind, empind, ny):
    x0, y0 = ind2xy(dupind, ny)
    x, y = ind2xy(empind, ny)
    dis = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return empind[np.argmin(dis)]

def ind2xy(ind, X):
    x = ind - (ind - 1) // X * X
    y = np.ceil(ind / X).astype(int)
    return x, y

def randp(P, sd, *args):
    np.random.seed(sd)
    X = np.random.rand(*args)
    P = P.flatten()
    if np.any(P < 0):
        raise ValueError('All probabilities should be 0 or larger.')
    if np.sum(P) == 0:
        X.fill(0)
    else:
        cum_prob = np.cumsum(P) / np.sum(P)
        X = np.searchsorted(cum_prob, X.flatten()).reshape(X.shape)
    return X

def ktUniformSampling(nx, ny, nt, ncalib, R):
    ptmp = np.zeros(ny)
    ptmp[np.round(np.arange(0, ny, R))] = 1
    ttmp = np.zeros(nt)
    ttmp[np.round(np.arange(0, nt , R))] = 1
    Top = toeplitz(ptmp, ttmp)
    ind = np.where(Top)
    ph = ind[0] - floor(ny / 2)
    ti = ind[1] - floor(nt / 2)

    ph, ti = ktdup(ph, ti, ny, nt)
    samp = np.zeros((ny, nt), dtype=int)
    ind = ((ph + floor(ny / 2)).astype(int), (ti + floor(nt / 2)).astype(int))
    samp[ind] = 1

    acs = zpad(np.ones((nx, ncalib, nt)), nx, ny, nt)
    ktus = np.transpose(np.tile(samp, (nx, 1, 1)), (0, 1, 2))
    mask_temp = ktus + acs
    mask = (mask_temp > 0).astype(int)
    return mask

def ktGaussianSampling(nx, ny, nt, ncalib, R, alpha, seed):
    p1 = np.arange(-floor(ny / 2), ceil(ny / 2))
    tr = round(ny / R)
    ti = np.full(tr * nt, 999999999, dtype=int)
    ph = np.zeros(tr * nt, dtype=int)
    sig = ny / 5
    prob = 0.1 + alpha / (1 - alpha + 1e-10) * np.exp(-(p1 ** 2) / (1 * sig ** 2))
    np.random.seed(seed)
    tmpSd = np.round(1e6 * np.random.rand(nt)).astype(int)
    ind = 0
    for i in range(-floor(nt / 2), ceil(nt / 2)):
        a = np.where(ti == i)[0]
        n_tmp = tr - len(a)
        prob_tmp = prob.copy()
        prob_tmp[a] = 0
        p_tmp = randp(prob_tmp, tmpSd[i + floor(nt / 2)], n_tmp)
        ti[ind:ind + n_tmp] = i
        ph[ind:ind + n_tmp] = p_tmp - floor(ny / 2) - 1
        ind += n_tmp

    ph, ti = ktdup(ph, ti, ny, nt)
    samp = np.zeros((ny, nt), dtype=int)
    ind = ((ph + floor(ny / 2)).astype(int), (ti + floor(nt / 2)).astype(int))
    samp[ind] = 1

    acs = zpad(np.ones((nx, ncalib, nt)), nx, ny, nt)
    ktus = np.transpose(np.tile(samp, (nx, 1, 1)), (0, 1, 2))
    mask_temp = ktus + acs
    mask = (mask_temp > 0).astype(int)
    return mask

def ktRadialSampling(nx, ny, nt, ncalib, R, angle4next, cropcorner):
    rate = 1 / R
    beams = floor(rate * 180)
    a = max(nx, ny) if cropcorner else ceil(np.sqrt(2) * max(nx, ny))
    aux = np.zeros((a, a))
    aux[round(a / 2), :] = 1
    angle = 180 / beams

    ktus = np.zeros((nx, ny, nt))
    for i in range(nt):
        angles = np.arange(0 + angle4next * i, 180 + angle4next * i, angle)
        image = np.zeros((nx, ny))
        for ang in angles:
            temp = crop(imrotate(aux, ang, 'constant'), nx, ny)
            image += (temp > 0.5)
            image = np.clip(image, 0, 1)
        ktus[:, :, i] = image

    acs = zpad(np.ones((ncalib, ncalib, nt)), nx, ny, nt)
    mask_temp = ktus + acs
    mask = (mask_temp > 0).astype(int)
    return mask

def ktMaskGenerator(nx, ny, nt, ncalib, R, pattern):
    if pattern == 'ktUniform':
        return ktUniformSampling(nx, ny, nt, ncalib, R)
    elif pattern == 'ktGaussian':
        alpha = 0.28 # default: 0.28, 0<alpha<1 controls sampling density; 0: uniform density, 1: maximally non-uniform density (Gaussian)
        seed = np.random.randint(1, 1001)
        return ktGaussianSampling(nx, ny, nt, ncalib, R, alpha, seed)
    elif pattern == 'ktRadial':
        angle4next = 137.5
        cropcorner = True
        R = R * 0.6
        return ktRadialSampling(nx, ny, nt, ncalib, R, angle4next, cropcorner)
    else:
        raise ValueError('No selected undersampling pattern. Please choose the proper one.')