#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
import logging
import sys
import time
import pytz
import csv
import pandas as pd
import numpy as np

from abc import ABC
from abc import abstractmethod
from logging.handlers import RotatingFileHandler
from logging import Logger

from typing import Optional
from typing import Dict
from typing import Any
from typing import List
from typing import Iterable

THRESHOLD_MISSING: int = 20

def pre_process_data(dfs: Iterable[pd.DataFrame]) -> List[pd.DataFrame]:
    global THRESHOLD_MISSING

    outputs: List[pd.DataFrame] = list()
    for df in dfs:
        # Fill NaN in categorical columns with 'NA'
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        df[cat_cols] = df[cat_cols].fillna('NA')
        # Remove rows with more than THRESHOLD_MISSING
        df = df[df.isna().sum(axis=1) < THRESHOLD_MISSING]
        # Reindex
        df.reset_index(drop=True, inplace=True)
        # Add the data frame to a processed list
        outputs.append(df)
    return outputs


class Method(ABC):

    def __init__(self, l: Logger, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params if params is not None else dict()
        self.logger = l

    @abstractmethod
    def train(self, x: pd.DataFrame) -> None:
        pass

    @staticmethod
    def predict(x: pd.DataFrame) -> pd.DataFrame:
        y = x[['ID']].copy()
        y['Pred'] = 0
        return y

class Random(Method):

    def __init__(self, l: Logger, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(l, params)

    def train(self, x: pd.DataFrame) -> None:
        self.logger.info("Random training started")
        self.logger.info(f"Random training parameters: {self.params}")
        self.logger.info("Random training ended")

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Random prediction started")
        y = super().predict(x)
        y['Pred'] = np.random.rand(len(y))
        self.logger.info("Random prediction ended")
        return y


if __name__ == "__main__":
    """ Main program to run the different ML approaches
    Should be called like:
    python ./src/solucions/exercise.py --train-file ./data/train.csv --test-file ./data/test.csv --result-file ./results/random.csv --method random
    """
    # Config the program arguments
    # noinspection DuplicatedCode
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train-file', type=str, help='Train CSV file', required=True)
    parser.add_argument('-s', '--test-file', type=str, help='Test CSV file', required=True)
    parser.add_argument('-r', '--result-file', type=str, help='Result CSV file to write', required=True)
    parser.add_argument('-m', '--method', type=str, help='Learning method to use', required=True, choices=['random','logistic'])
    parser.add_argument('-l', '--log-file', type=str, help='File to log progress and errors', required=False)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    if args.log_file is not None:
        handler = RotatingFileHandler(args.log_file, mode='a', maxBytes=5*1024*1024, backupCount=15, encoding='utf-8', delay=False)
        logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s', handlers=[handler], encoding='utf-8', level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")
    else:
        handler = ch = logging.StreamHandler()
        logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s', handlers=[handler], encoding='utf-8', level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")

    # Process the CSV file and store it into the database
    logger.info("Loading train set")
    train_df = pd.read_csv(args.train_file)
    logger.info("Loading test set")
    test_df = pd.read_csv(args.test_file)

    train_df, test_df = pre_process_data((train_df, test_df))

    result: pd.DataFrame = pd.DataFrame()
    if args.method == 'random':
        method = Random(logger)
        method.train(train_df)
        result = method.predict(test_df)

    result.to_csv(args.result_file, index=False)
