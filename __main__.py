"""
    This module implements the command-line interface.
"""
import logging

from dataset import Dataset
from evaluation import evaluate_task1
from filenames import DATASET_DIRNAME, LOG_FILENAME

def main():
    """ Runs the main evaluation routine. """
    dataset = Dataset(DATASET_DIRNAME)
    evaluate_task1(dataset.task1_evaluation_dataset())

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(name)s | %(levelname)s : %(message)s',
                        filename=LOG_FILENAME, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    main()
