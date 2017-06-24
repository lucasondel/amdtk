
"""Acoustic Model Discovery Toolkit (AMDTK) module.
Set of tools to do Bayesian clustering of raw acoustic
features to automatically discover phone-like units.

"""

from .mixture import Mixture
from .mixture import DiscriminativeMixture
from .phone_loop import PhoneLoop
from .vae import MLPClassifier, SVAE

