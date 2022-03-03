"""
Estimates a linear warp field, the target is a transformed version of lenna:

    http://en.wikipedia.org/wiki/Lenna
"""

import scipy.ndimage as nd
import scipy.misc as misc

from imreg import register, model, metric
from imreg.samplers import sampler

import matplotlib.pyplot as plt

# Form some test data (lena, lena rotated 20 degrees)
image = plt.imread('data/cameraman.png')
template = nd.rotate(image, 20, reshape=False)

# Form the affine registration instance.
affine = register.Register(
    model.Affine,
    metric.Residual,
    sampler.Spline
    )

# Coerce the image data into RegisterData.
image = register.RegisterData(image).downsample(2)
template = register.RegisterData(template).downsample(2)

# Register.
step, search = affine.register(
    image,
    template,
    verbose=True,
    )
