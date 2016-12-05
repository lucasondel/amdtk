
"""Acoustic Model Discovery Toolkit (AMDTK) module.
Set of tools to do Bayesian clustering of raw acoustic
features to automatically discover phone-like units.

"""
from .internal_io import read_htk
from .internal_io import write_htk
from .internal_io import read_htk_labels
from .internal_io import write_htk_labels
from .internal_io import read_timit_labels
from .internal_io import read_mlf
from .internal_io import write_mlf
from .internal_io import read_ctm
from .internal_io import write_eval_to_clusters
