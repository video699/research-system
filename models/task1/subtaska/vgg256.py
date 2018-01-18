"""
    This module implements an unsupervised task 1, subtask A model that uses an image classification
    convolutional neural network based on the VGG-256 architecture (Simonyan and Zisserman, 2015).
    Pairs of images are ranked based on the cosine distances between the images.
"""

from scipy.spatial.distance import cosine

from .base import Model

DATASET_NAMES = ["ImageNet", "ImageNet+Places2"]

class VGG256(Model):
    """
        This class represents a task 1, subtask A model that uses an image classification
        convolutional neural network based on the VGG-256 architecture (Simonyan and Zisserman,
        2015). Pairs of images are ranked based on the cosine distances between the images.
    """
    def __init__(self, vgg_datasets="ImageNet+Places2", be_handicapped=False):
        """Constructs an unsupervised task1, subtask A model that uses an image classification
        convolutional neural network based on the VGG-256 architecture (Simonyan and Zisserman,
        2015).

        Parameters:
            vgg_datasets    The dataset on which the network should be trained. The permissible
                            values include "ImageNet" for a network trained on the ImageNet dataset
                            (http://www.image-net.org/) and "ImageNet+Places2" for a network trained
                            on both the ImageNet and Places2 (http://places2.csail.mit.edu/)
                            datasets.

            be_handicapped  Whether the classifier should handicap itself by using the feature (True)
                            vectors computed from the full video frames rather than feature vectors
                            computed from the cropped screens (False).
        """
        assert vgg_datasets in DATASET_NAMES
        self.vgg_datasets = vgg_datasets
        self.dataset_index = 0 if self.vgg_datasets == "ImageNet" else 1
        assert type(be_handicapped) is bool
        self.is_handicapped = be_handicapped

    def predict(self, observations):
        rankings = []
        for screen_video, page_video in observations:
            pages = page_video.pages
            for screen in screen_video.screens:
                screen_vgg256 = screen.frame.vgg256 if self.is_handicapped else screen.vgg256
                ranking = [1-cosine(page.vgg256[self.dataset_index],
                                    screen_vgg256[self.dataset_index]) for page in pages]
                rankings.append(ranking)
        return rankings

    def _filename(self):
        return "%s.%s-%s-%s" % (__name__, self.__class__.__name__, self.vgg_datasets, \
                                self.is_handicapped)

    def __repr__(self):
        return "VGG256 (%s, %shandicapped)" % (self.vgg_datasets,
                                               "" if self.is_handicapped else "not ")

VGGS256 = [VGG256(dataset, is_handicapped) \
           for dataset in DATASET_NAMES \
           for is_handicapped in (True, False)]
