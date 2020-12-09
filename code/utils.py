import numpy as np

def get_lightcurve(tic):
    data = np.loadtxt('%i.txt' % tic)
    flag = data[:,6]
    idx = (flag == 0) & (~np.isnan(data[:,0])) & (~np.isnan(data[:,1]))
    flux = data[:,2][idx]
    sigma = 1.4826*np.nanmedian(np.abs(flux - np.nanmedian(flux)))
    outlier = np.nanmedian(flux) + 5*sigma
    out = (flux > outlier)
    time = data[:,0][idx][~out]
    raw = data[:,1][idx][~out]
    flux = data[:,2][idx][~out]
    flux1 = data[:,3][idx][~out]
    flux2 = data[:,4][idx][~out]
    flux3 = data[:,5][idx][~out]
    return time, raw, flux, flux1, flux2, flux3, flag

