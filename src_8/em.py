# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.patches import Ellipse

#混合ガウス分布  par = (pi, mean, var): (混合係数、平均、分散)
def gaussians(x, par):
    return [gaussian(x-mu, var) * pi for pi,mu,var in zip(*par)]

#ガウス分布
def gaussian(x, var):
    nvar = n_variate(x)
    if not nvar:
        qf, detvar, nvar = x**2/var, var, 1
    else:
        qf, detvar = np.dot(np.linalg.solve(var, x), x), np.linalg.det(var)
    return np.exp(-qf/2) / np.sqrt(detvar*(2*np.pi)**nvar)

#対数尤度
def loglikelihood(data, par):
    gam = [gaussians(x, par) for x in data]
    ll = sum([np.log(sum(g)) for g in gam])
    return ll, gam

#Eステップ
def e_step(data, pars):
    ll, gam = loglikelihood(data, pars)
    print('gam')
    print(gam)
    gammas = transpose(list(map(normalize, gam)))
    return gammas, ll

#Mステップ  pars = (pis, means, vars)
def m_step(data, gammas):
    ws = list(map(sum, gammas))
    pis = normalize(ws)
    means = [np.dot(g, data)/w for g, w in zip(gammas, ws)]
    sqr_data = list(map(lambda x:x**2, data))
    #vars = [np.dot(g, sqr_data)/w - (np.dot(g, data)/w)**2 for g, w in zip(gammas, ws)]
    vars = [make_var(g, data, mu)/w for g, w, mu in zip(gammas, ws, means)]
    return pis, means, vars

#共分散
def make_var(gammas, data, mean):
    return np.sum([g * make_cov(x-mean) for g, x in zip(gammas, data)], axis=0)

def make_cov(x):
    nvar = n_variate(x)
    if not nvar:
        return x**2
    m = np.matrix(x)
    return m.reshape(nvar, 1) * m.reshape(1, nvar)

#n-変量
def n_variate(x):
    if isinstance(x, (list, np.ndarray)):
        return len(x)
    return 0  # univariate

#正規化
def normalize(lst):
    s = sum(lst)
    return [x/s for x in lst]

#転置
def transpose(a):
    return list(map(list,zip(*a)))

def flatten(lst):
    if isinstance(lst[0], np.ndarray):
        lst = list(map(list, lst))
    return sum(lst, [])

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]


#混合ガウス分布データ（K: 混合ガウス分布の数）
def make_data(typ_nvariate):
    if typ_nvariate == 'univariate':  # 単変量
        par = [[2.0, 0.2, 300], [4.0, 0.4, 600], [6.0, 0.4, 100]]
        data = flatten([np.random.normal(mu,sig,n) for mu,sig,n in par])
        K = len(par)
        means = [np.random.choice(data) for _ in range(K)]
        vars = [np.var(data)]*K
    elif typ_nvariate == 'bivariate':  # 2変量
        nvar, ndat, sig = 2, 250, 0.4
        centers = [[1, 1], [-1, -1], [1, -1]]
        K = len(centers)
        data = flatten([np.random.randn(ndat,nvar)*sig + np.array(c) for c in centers])
        means = np.random.rand(K, nvar)
        vars = [np.identity(nvar)]*K
    pis = [1.0/K]*K
    return data, [pis, means, vars]

#EMアルゴリズム（gammas: 'burden rates', or 'responsibilities'）
def em(typ_nvariate='univariate'):
    delta_ls, max_step = 1e-5, 400
    lls, pars = [], []  #各ステップの計算結果を保存
    data, par = make_data(typ_nvariate)
    for i in range(max_step):
        gammas, ll = e_step(data, par)
        par = m_step(data, gammas)
        pars.append(par)
        lls.append(ll)
        print('step'+str(i))
        print(par)
        if len(lls) > 8 and lls[-1] - lls[-2] < delta_ls:
            break
    # 結果出力
    print('nstep=%3d' % len(lls), " log(likelihood) =", lls[-1])
    return 0

def main():
    em("univariate")
if __name__ == '__main__':
    main()
