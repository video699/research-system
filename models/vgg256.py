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
    def __init__(self, vgg_datasets="ImageNet+Places2", be_handycapped=False):
        """Constructs a task1, subtask A model that uses an image classification convolutional
        neural network based on the VGG-256 architecture (Simonyan and Zisserman, 2015).

        Parameters:
            vgg_datasets    The dataset on which the network should be trained. The permissible
                            values include "ImageNet" for a network trained on the ImageNet dataset
                            (http://www.image-net.org/) and "ImageNet+Places2" for a network trained
                            on both the ImageNet and Places2 (http://places2.csail.mit.edu/)
                            datasets.

            be_handycapped  Whether the classifier should handycap itself by using the feature (True)
                            vectors computed from the full video frames rather than feature vectors
                            computed from the cropped screens (False).
        """
        assert vgg_datasets in ("ImageNet", "ImageNet+Places2")
        self.vgg_datasets = vgg_datasets
        self.dataset_index = 0 if self.vgg_datasets == "ImageNet" else 1
        assert type(be_handycapped) is bool
        self.is_handycapped = be_handycapped

    def predict(self, observations):
        for screen, pages in observations:
            screen_vgg256 = screen.frame.vgg256 if self.is_handycapped else screen.vgg256
            ranking = [1-cosine(page.vgg256[self.dataset_index],
                                screen_vgg256[self.dataset_index]) for page in pages]
            yield ranking

    def _filename(self):
        return "%s.%s.%s-%s-%s" % (super(VGG256Model, self)._filename(), __name__, \
                                   self.__class__.__name__, self.vgg_datasets, self.is_handycapped)

    def __repr__(self):
        return "VGG256 (%s, %shandycapped)" % (self.vgg_datasets,
                                               "" if self.is_handycapped else "not ")

VGG256_IMAGENET = VGG256Model("ImageNet", False)
VGG256_IMAGENET_HANDYCAPPED = VGG256Model("ImageNet", True)
VGG256_IMAGENET_PLACES2 = VGG256Model("ImageNet+Places2", False)
VGG256_IMAGENET_PLACES2_HANDYCAPPED = VGG256Model("ImageNet+Places2", True)
