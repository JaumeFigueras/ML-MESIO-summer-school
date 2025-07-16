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
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoost
from catboost import Pool

from typing import Optional
from typing import Dict
from typing import Any
from typing import List
from typing import Iterable
from typing import Tuple

THRESHOLD_MISSING: int = 20
THRESHOLD_CORRELATION: float = 0.99
TEST_TRAIN_RATIO: float = 0.3
RANDOM_SEED: int = 1234
EARLY_STOPPING_ROUNDS = 100

def pre_process_data(dfs: Iterable[pd.DataFrame]) -> List[pd.DataFrame]:
    global THRESHOLD_MISSING

    outputs: List[pd.DataFrame] = list()
    for i, df in enumerate(dfs):
        # Fill NaN in categorical columns with 'NA'
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        df[cat_cols] = df[cat_cols].fillna('NA')
        # Remove rows with more than THRESHOLD_MISSING
        if i == 0:
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
    def train(self, x: pd.DataFrame, y:pd.DataFrame) -> None:
        pass

    @staticmethod
    def predict(x: pd.DataFrame) -> pd.DataFrame:
        y = x[['ID']].copy()
        y['Pred'] = 0
        return y

    def analyse_correlation(self, df: pd.DataFrame, correlation_threshold: float) -> List[str]:
        # get the Pearson correlation matrix
        numerical_df = df.select_dtypes(exclude=['object', 'category'])
        correlation_matrix = numerical_df.corr().abs()  # Correlation Matrix
        # Extract the upper triangular excluding the diagonal and marking True to the elements that we like to analyse and
        # NaN to those wee don't like to analyse. Where is equivalent to a & mask
        upper_matrix = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        correlated_columns: List[str] = [col for col in upper_matrix.columns if
                                         any(upper_matrix[col] > correlation_threshold)]
        self.logger.info(f"Correlated columns: {', '.join(correlated_columns) if len(correlated_columns) else 'None'}")
        return correlated_columns

    def analyse_constant(self, df: pd.DataFrame) -> List[str]:
        constant_columns =  [col for col in df.columns if df[col].nunique() < 2]
        self.logger.info(f"Constant columns: {', '.join(constant_columns) if len(constant_columns) > 0 else 'None'}")
        return constant_columns

class Random(Method):

    def __init__(self, l: Logger, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(l, params)

    def train(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.logger.info("Random training started")
        self.logger.info(f"Random training parameters: {self.params}")
        self.logger.info("Random training ended")

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Random prediction started")
        y = super().predict(x)
        y['Pred'] = np.random.rand(len(y))
        self.logger.info("Random prediction ended")
        return y

class Catboost(Method):
    def __init__(self, l: Logger, model_params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(l, model_params)
        self.model_catboost = CatBoostClassifier(**self.params)

    def process_data(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.logger.info("Processing data")
        self.logger.info(f"Processing Correlation")
        correlated_columns = self.analyse_correlation(train, THRESHOLD_CORRELATION)
        processed_train = train.drop(columns=correlated_columns)
        processed_test = test.drop(columns=correlated_columns)
        self.logger.info(f"Processing Constant columns")
        constant_columns = self.analyse_constant(processed_train)
        processed_train.drop(columns=constant_columns, inplace=True)
        processed_test.drop(columns=constant_columns, inplace=True)
        return processed_train, processed_test


    def train(self, x: pd.DataFrame, y: pd.DataFrame, run_params: Optional[Dict[str, Any]] = None) -> None:
        self.logger.info("Catboost training started")
        self.logger.info(f"Catboost training parameters: {self.params}")
        categorical: List[str] = list(x.select_dtypes(include=['object', 'category']).columns)
        x_to_train, x_to_validate, y_to_train, y_to_validate = train_test_split(x, y, test_size=TEST_TRAIN_RATIO, random_state=RANDOM_SEED)
        pool_tr = Pool(x_to_train, y_to_train, cat_features=categorical)
        pool_val = Pool(x_to_validate, y_to_validate, cat_features=categorical)
        self.model_catboost.set_params(**run_params)
        self.model_catboost.fit(X=pool_tr, eval_set=pool_val, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        self.logger.info("Catboost training ended")

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Catboost prediction started")
        y = super().predict(x)
        y['Pred'] = self.model_catboost.predict_proba(x.drop(columns=['ID']))[:, 1]
        self.logger.info("Catboost prediction ended")
        return y

class CatboostKFold(CatBoost):


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
    parser.add_argument('-m', '--method', type=str, help='Learning method to use', required=True, choices=['random','catboost'])
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
        method.train(train_df, None)
        result = method.predict(test_df)
    if args.method == 'catboost':
        method = Catboost(logger, model_params={'eval_metric': 'AUC',
                                                'iterations': 5000,
                                                'od_type': 'Iter',
                                                'random_seed': RANDOM_SEED,
                                                'verbose': 50})
        train_df, test_df = method.process_data(train_df, test_df)
        x_train = train_df.drop(columns=['ID', 'TARGET']).reset_index(drop=True)
        y_train = train_df['TARGET'].reset_index(drop=True)
        x_test = test_df.reset_index(drop=True)
        method.train(x_train, y_train, run_params={'objective': 'Logloss',
                                                   'learning_rate': 0.01,  # learning rate, lower -> slower but better prediction
                                                   'depth': 4,  # Depth of the trees (values betwwen 5 and 10, higher -> more overfitting)
                                                   'min_data_in_leaf': 150,
                                                   'l2_leaf_reg': 20,  # L2 regularization (between 3 and 20, higher -> less overfitting)
                                                   'rsm': 0.5, # % of features to consider in each split (lower -> faster and reduces overfitting)
                                                   'subsample': 0.5,  # Sample rate for bagging
                                                   'random_seed': RANDOM_SEED
                                                   })
        result = method.predict(x_test)

    result.to_csv(args.result_file, index=False)
