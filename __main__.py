"""
    This module implements the command-line interface.
"""
import logging

from dataset import Dataset
from evaluation import evaluate_task1
from filenames import DATASET_DIRNAME, LOG_FILENAME
from preprocessing import load_caches, dump_caches

def main():
    """ Runs the main evaluation routine. """
#   load_caches()
    dataset = Dataset(DATASET_DIRNAME)
    evaluate_task1(dataset.task1_evaluation_dataset())
#   dump_caches()

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(name)s | %(levelname)s : %(message)s',
                        filename=LOG_FILENAME, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    main()
