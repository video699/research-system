"""
    This module implements a task 1, subtask A model that uses an image classification convolutional
    neural network based on the VGG-256 architecture (Simonyan and Zisserman, 2015).
"""

from scipy.spatial.distance import cosine

from .base import Task1SubtaskAModel

class VGG256Model(Task1SubtaskAModel):
    """
        This class represents a task 1, subtask A model that uses an image classification
        convolutional neural network based on the VGG-256 architecture (Simonyan and Zisserman,
        2015).
    """
    def __init__(self, vgg_datasets="ImageNet+Places2"):
        """Constructs a task1, subtask A model that uses an image classification convolutional
        neural network based on the VGG-256 architecture (Simonyan and Zisserman, 2015).

        Parameters:
            vgg_datasets    The dataset on which the network should be trained. The permissible
                            values include "ImageNet" for a network trained on the ImageNet dataset
                            (http://www.image-net.org/) and "ImageNet+Places2" for a network trained
                            on both the ImageNet and Places2 (http://places2.csail.mit.edu/)
                            datasets.
        """
        assert vgg_datasets in ("ImageNet", "ImageNet+Places2")
        self.vgg_datasets = vgg_datasets
        self.dataset_index = 0 if self.vgg_datasets == "ImageNet" else 1

    def predict(self, observations):
        for screen, pages in observations:
            ranking = [1-cosine(page.vgg256[self.dataset_index],
                                screen.vgg256[self.dataset_index]) for page in pages]
            yield ranking

    def __repr__(self):
        return "VGG256 (%s)" % self.vgg_datasets

VGG256_IMAGENET = VGG256Model("ImageNet")
VGG256_IMAGENET_PLACES2 = VGG256Model("ImageNet+Places2")
