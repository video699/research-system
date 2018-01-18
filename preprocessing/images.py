"""
    This module contains the image processing routines.
"""

import logging
import lzma
from lzma import FORMAT_XZ
from math import log2
from pickle import load, dump

import cv2
from mahotas.features import haralick
from skimage.morphology import reconstruction
import numpy as np

from dataset import Frame, Screen, Page, Video, crop
from filenames import IMAGE_CACHES_FILENAME

LOGGER = logging.getLogger(__name__)

MORPH_SE = cv2.MORPH_RECT
MORPH_SE_DEFAULT_SIZE = 65
WIDTH = 512
HEIGHT = 512
BW_BG_MIN_RATIO = 1.8
BW_MIN_COMPONENT_DIAMETER = 3
BW_BORDER_WIDTH = WIDTH // 40
BW_BORDER_HEIGHT = HEIGHT // 40
BW_THRESHOLD = cv2.THRESH_BINARY | cv2.THRESH_OTSU
RESIZE_INTERPOLATION = cv2.INTER_AREA
HISTOGRAM_NUM_BINS = 16

CACHES = {
    "grayscale": {},
    "bw": {},
    "variance": {},
    "row": {},
    "histogram": {},
    "haralick": {},
}

def load_caches():
    """
        Loads the image caches off a persistent storage.
    """
    global CACHES
    try:
        LOGGER.debug("Loading the image caches ...")
        with lzma.open("%s.pkl.xz" % IMAGE_CACHES_FILENAME, "rb") as f:
            CACHES = load(f)
        LOGGER.debug("Done loading the image caches.")
    except FileNotFoundError:
        LOGGER.debug("Image caches not found.")

def dump_caches():
    """
        Dumps the image caches on a persistent storage.
    """
    LOGGER.debug("Dumping the image caches ...")
    with open("%s.pkl.xz" % IMAGE_CACHES_FILENAME, "wb", format=FORMAT_XZ, preset=9) as f:
        dump(CACHES, f)
    LOGGER.debug("Done dumping the image caches.")

class Image(object):
    """
        This class represents an image corresponding to the provided Frame, Screen, or Page object.
    """
    def __init__(self, obj, illumination=("closing", MORPH_SE_DEFAULT_SIZE),
                 pixel_weights=("rows",), flipping=True):
        """Constructs an image object.

        Parameters:
            obj             The provided Frame, Screen, or Page object.
            illumination    Our model of illumination. The admissible values are:
                            (i) None -- We do not model illumination.
                            (ii) ("closing", size) -- We use the morphological closing with a
                                 structuring element of the given size as a model of illumination.
            pixel_weights   How we weight the individual pixels when we are computing a grayscale
                            histogram. The admissible values are:
                            (i) set() -- We weight the pixels uniformly.
                            (ii) set(("var",)) -- We weight the pixels by their variance across the
                                 video.
                            (iii) set(("rows",)) -- We only consider pixels that fall into the text
                                  rows in the image.
                            (iv) set(("rows", "var")) -- We only consider pixels that fall into the
                                 text rows in the image and we weight the pixels by their variance.
            flipping        Whether we flip images in which the background is dark and the text is
                            light.
        """
        assert isinstance(obj, Frame) or isinstance(obj, Screen) or isinstance(obj, Page)
        self.obj = obj

        assert illumination is None or illumination[0] == "closing"
        if isinstance(obj, Screen):
            self.illumination = illumination
        else:
            self.illumination = None

        pixel_weights = set(pixel_weights)
        for value in pixel_weights:
            assert value in ("var", "rows")
        self.pixel_weights = pixel_weights

        assert flipping in (True, False)
        if isinstance(obj, Frame):
            self.flipping = False
        else:
            self.flipping = flipping
            
    def get_image(self):
        """
            Returns a grayscale version of the image.
        """
        if (self.obj, self.illumination, self.flipping) not in CACHES["grayscale"]:
            if isinstance(self.obj, Screen):
                # Crop out the screen.
                image_bgr_frame = cv2.imread(self.obj.frame.filename)
                image_bgr = crop(image_bgr_frame, self.obj.bounds)
            else:
                image_bgr = cv2.imread(self.obj.filename)
            image_gray_uneven = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            del image_bgr
            if self.flipping:
                # Detect the polarity of the image.
                bwimage = cv2.threshold(image_gray_uneven, 0, 255, BW_THRESHOLD)[1]
                fg = np.sum(bwimage == 0)
                bg = np.sum(bwimage == 255)
                del bwimage
                if bg / fg < BW_BG_MIN_RATIO:
                    image_gray_uneven = np.max(image_gray_uneven) - image_gray_uneven
            if self.illumination is not None:
                # Correct uneven illumination.
                illumination_model, se_size = self.illumination
                assert se_size % 2 == 1
                selem = cv2.getStructuringElement(MORPH_SE, (se_size, se_size))
                if illumination_model == "closing":
                    closing = cv2.morphologyEx(image_gray_uneven, cv2.MORPH_CLOSE, selem)
                    image_gray = np.max(image_gray_uneven) - (closing - image_gray_uneven)
                    del closing
            else:
                image_gray = image_gray_uneven
            del image_gray_uneven
            image_stretched = (image_gray - np.min(image_gray)) / \
                              (np.max(image_gray) - np.min(image_gray))
            del image_gray
            image_resized = cv2.resize(image_stretched, (WIDTH, HEIGHT),
                                       interpolation=RESIZE_INTERPOLATION).clip(0)
            CACHES["grayscale"][(self.obj, self.illumination, self.flipping)] = image_resized
        return CACHES["grayscale"][(self.obj, self.illumination, self.flipping)]

    def get_haralick(self):
        """
            Returns a vector of Haralick features for the image.
        """
        if (self.obj, self.illumination, self.flipping) not in CACHES["haralick"]:
            image = (self.get_image() * 255).astype("uint8")
            CACHES["haralick"][(self.obj, self.illumination, self.flipping)] = haralick(image)
        return CACHES["haralick"][(self.obj, self.illumination, self.flipping)]

    def get_bwimage(self):
        """
            Returns a monochrome version of the image.
        """
        if (self.obj, self.illumination, self.flipping) not in CACHES["bw"]:
            image = self.get_image()
            image_uint8 = (image * 255).astype("uint8")
            del image
            bwimage = 255 - cv2.threshold(image_uint8, 0, 255, BW_THRESHOLD)[1]
            del image_uint8
            if isinstance(self.obj, Screen):
                # Clear the noise from the image.
                bwimage_eroded = cv2.morphologyEx(bwimage, cv2.MORPH_ERODE,\
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BW_MIN_COMPONENT_DIAMETER,
                                                                  BW_MIN_COMPONENT_DIAMETER)))
                bwimage_denoised = reconstruction(bwimage_eroded, bwimage,
                                                  "dilation").astype("uint8")
                del bwimage, bwimage_eroded
            else:
                bwimage_denoised = bwimage
            # Remove the objects touching borders.
            border_marker = np.array(bwimage_denoised)
            border_marker[BW_BORDER_WIDTH-1:-BW_BORDER_WIDTH,
                          BW_BORDER_HEIGHT-1:-BW_BORDER_HEIGHT] = 0
            border_components = reconstruction(border_marker, bwimage_denoised,
                                               "dilation").astype("uint8")
            del border_marker
            bwimage_no_borders = bwimage_denoised - border_components
            del bwimage_denoised, border_components
            CACHES["bw"][(self.obj, self.illumination, self.flipping)] = bwimage_no_borders
        return CACHES["bw"][(self.obj, self.illumination, self.flipping)]

    def get_rows(self):
        """
            Returns the number, the heights, the altitude of the individual text rows in the image,
            the derived histograms, and the rows themselves.
        """
        if (self.obj, self.illumination, self.flipping) not in CACHES["row"]:
            bwimage = self.get_bwimage()
            selem = np.ones((1, WIDTH*2+1), dtype="uint8")
            row_image = cv2.morphologyEx(bwimage, cv2.MORPH_DILATE, selem)
            del bwimage, selem
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(row_image)
            num_rows = num_labels - 1
            row_heights = stats[1:,4] // WIDTH
            row_altitudes = centroids[1:,1]
            del num_labels, labels, stats, centroids
            height_histogram, height_histogram_bins = \
                np.histogram(row_heights, bins=HISTOGRAM_NUM_BINS, range=(1, HEIGHT))
            altitude_histogram, altitude_histogram_bins = \
                np.histogram(row_altitudes, bins=HISTOGRAM_NUM_BINS, range=(0, HEIGHT - 1))
            rows = (num_rows, row_heights, row_altitudes,
                    ((height_histogram, height_histogram_bins),
                     (altitude_histogram, altitude_histogram_bins)), row_image)
            CACHES["row"][(self.obj, self.illumination, self.flipping)] = rows
        return CACHES["row"][(self.obj, self.illumination, self.flipping)]

    def get_histogram(self):
        """
            Returns the grayscale histogram of the image.
        """
        if self not in CACHES["histogram"]:
            image = self.get_image()
            weights = np.ones_like(image)
            if "rows" in self.pixel_weights:
                weights = self.get_rows()[4]
            if "var" in self.pixel_weights:
                variance = get_pixel_variance(self.obj.video)
                weights = weights.astype("float64") * variance
                del variance
            histogram = np.histogram(image, bins=HISTOGRAM_NUM_BINS, range=(0, 1),
                                     weights=weights)
            CACHES["histogram"][self] = histogram
        return CACHES["histogram"][self]

    def __hash__(self):
        return (self.obj, self.illumination, self.flipping,
                "rows" in self.pixel_weights,
                "var" in self.pixel_weights).__hash__()

    def __eq__(self, other):
        return isinstance(other, Image) and self.obj == other.obj \
            and self.illumination == other.illumination and self.flipping == other.flipping \
            and self.pixel_weights == self.pixel_weights

    def __repr__(self):
        return "Image of %s (illumination: %s, pixel weights: %s, flipping: %s)" \
            % (self.obj, self.illumination, self.pixel_weights, self.flipping)

def get_pixel_variance(video):
    """Returns the grayscale intensity variance of pixels in the document pages of provided video.

    Parameters:
        video   The provided video."""
    assert isinstance(video, Video)
    if video not in CACHES["variance"]:
        images = [Image(page).get_image() for page in video.pages]
        variance = np.var(images, axis=0)
        CACHES["variance"][video] = variance
    return CACHES["variance"][video]

# Tunable parameters for the preprocessing of images.
KWARGS = [{"illumination": illumination,
           "pixel_weights": pixel_weights,
           "flipping": flipping} \
          for illumination in ([None] + [("closing", 2**(selem_size+1)+1) \
                                         for selem_size in range(int(log2(min(WIDTH, HEIGHT))))]) \
          for pixel_weights in [(), ("rows",), ("var",), ("rows", "var")] \
          for flipping in (True, False)]
