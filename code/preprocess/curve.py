from scipy import interpolate
import utils.general as utils
import numpy as np
import os

homedir = os.environ['HOME']
root = '{0}/data/datasets/toy_curve/'.format(homedir)


output_path = root
utils.mkdir_ifnotexists(output_path)



t = np.array([[-0.91899198, -0.80607603, -0.18083174, 0.2868784, -0.86806828,
                       0.56029102, -0.98040225, -0.91899198],
                      [0.58470931, 0.34454382, -0.08756032, 0.59659409, 0.89598192,
                       0.34345276, -0.3451372, 0.58470931],
                      [0.29763193, -0.26568314, -0.64020486, -0.99736097, 0.85872213,
                       -0.19288592, 0.08070222, 0.29763193]])
npoints = 100

tck, u = interpolate.splprep([t[0, :], t[1, :], t[2, :]], s=0, per=True)
x, y, z = interpolate.splev(np.linspace(0, 1, npoints * 5), tck)
samples = np.vstack([x, y, z]).transpose()
samples -= samples.mean(axis=0)
#samples = samples[np.newaxis, :]

idx = utils.fps(samples, npoints)
points = samples[idx]

noise_sigma = 0.01

points = (points + np.random.normal(0, noise_sigma, size=points.shape)).astype(np.float32)
file = 'toy_curve.ply'

np.save(os.path.join(output_path,file),points)

from utils.plots import plot_threed_scatter
plot_threed_scatter(points,'.',0,0)