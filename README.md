To run the evaluation,

- download the dataset:
    ```sh
    $ git submodule update --init
    ```
- set up a Python 3 virtual environment and install the required packages:
    ```sh
    $ mkvirtualenv -p $(which python3) system
    (system) $ pip install -U pip
    (system) $ pip install -r requirements.txt
    (system) $ pip install -r dataset/requirements.txt
    ```
- remove the log file with results:
    ```sh
    (system) $ rm __main__.log
    ```
- run the main shell script:
    ```sh
    (system) $ ./__main.sh
    ```

Evaluation results will be printed to the standard output and stored inside the
`__main__.log` log file. The dataset takes up about 579M of disk space. The
evaluation requires about 70G of memory and a month of wall clock time with 32
Intel Xeon E5-2650 v2 (2.60 GHz) CPU cores; you can reduce the memory
requirements by turning the individual `preprocessing.images.CACHES` and
`preprocessing.features.CACHES` dictionaries into LRU caches with fixed size.
