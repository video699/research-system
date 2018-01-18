"""
    This module provides data preprocessing capabilities.
"""
from .images import WIDTH as IMAGE_WIDTH, HEIGHT as IMAGE_HEIGHT, KWARGS as IMAGE_KWARGS, Image, \
    load_caches as load_image_caches, dump_caches as dump_image_caches
from .features import KWARGS as FEATURE_KWARGS, Features, load_caches as load_feature_caches, \
    dump_caches as dump_feature_caches

def load_caches():
    """
        Loads the preprocessing caches off a persistent storage.
    """
    load_image_caches()
    load_feature_caches()

def dump_caches():
    """
        Dumps the preprocessing caches on a persistent storage.
    """
    dump_image_caches()
    dump_feature_caches()
