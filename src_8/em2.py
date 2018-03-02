import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import datasets
from sklearn import mixture


def scale(X):
    """データ行列Xを属性ごとに標準化したデータを返す"""
    # 属性の数（=列の数）
    col = X.shape[1]

    # 属性ごとに平均値と標準偏差を計算
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    # 属性ごとデータを標準化
    for i in range(col):
        X[:,i] = (X[:,i] - mu[i]) / sigma[i]

    return X

def flatten(lst):
    if isinstance(lst[0], np.ndarray):
        lst = list(map(list, lst))
    return sum(lst, [])

if __name__ == '__main__':
  gmm=mixture.GMM(n_components=3, covariance_type='full', n_iter=1000)
  par = [[2.0, 0.2, 300], [4.0, 0.4, 600], [6.0, 0.4, 100]]
  data = flatten([np.random.normal(mu,sig,n) for mu,sig,n in par])
  gaiji = list(map(lambda x:[x],data))
  #x_train = scale(ar_data)
  gmm.fit(gaiji)
  # 結果を表示
  print('*** weights')
  print(str(gmm.weights_))

  print('*** means')
  print(str(gmm.means_))

  print('*** covars')
  print(listr(gmm.covars_))
