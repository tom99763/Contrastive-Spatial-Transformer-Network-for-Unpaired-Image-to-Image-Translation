import numpy as np


def calculate_fid(Eb, Eab):
    # calculate mean and covariance statistics
    mu1, sigma1 = Eb.mean(axis=0), np.cov(Eb, rowvar=False)
    mu2, sigma2 = Eab.mean(axis=0), np.cov(Eab, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))

    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
