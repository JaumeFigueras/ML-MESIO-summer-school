#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import pandas as pd
import numpy as np

from abc import ABC
from abc import abstractmethod
from logging.handlers import RotatingFileHandler
from logging import Logger
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
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
EARLY_STOPPING_ROUNDS: int = 100
NUMBER_OF_FOLDS: int = 5

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
    def __init__(self, l: Logger, model_params: Optional[Dict[str, Any]] = None, random_seed: Optional[int] = None) -> None:
        super().__init__(l, model_params)
        self.model_catboost = CatBoostClassifier(**self.params)
        self.random_seed = random_seed

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

class CatboostKFold(Catboost):
    def __init__(self, l: Logger, model_params: Optional[Dict[str, Any]] = None, random_seed: Optional[int] = None) -> None:
        super().__init__(l, model_params, random_seed)

    def train_predict(self, number_of_folds: int, random_seed: int,  make_prediction: bool, x: pd.DataFrame, y: pd.DataFrame, x_test: pd.DataFrame, run_params: Dict[str, Any]) -> Any:
        # Create the k folds
        kf = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=self.random_seed)
        train_df_level_1 = pd.DataFrame(np.zeros((x.shape[0], 1)), columns=['train_y_hat'])
        test_df_level_1 = pd.DataFrame()
        self.model_catboost.set_params(**run_params)
        for i, (train_index, test_index) in enumerate(kf.split(x, y)):
            fold_x_train = x.loc[train_index.tolist(), :]
            fold_x_test = x.loc[test_index.tolist(), :]
            fold_y_train = y[train_index.tolist()]
            fold_y_test = y[test_index.tolist()]
            categorical: List[str] = list(x.select_dtypes(include=['object', 'category']).columns)
            if len(categorical) > 0:
                # Prepare Pool
                pool_train = Pool(fold_x_train, fold_y_train, cat_features=categorical)
                # (k-1)-folds model adjusting

                self.model_catboost.fit(X=pool_train)
            else:
                # (k-1)-folds model adjusting

                self.model_catboost.fit(fold_x_train, fold_y_train)
            # Predict on the free fold to evaluate metric
            # and on train to have an overfitting-free prediction for the next level
            p_fold = self.model_catboost.predict_proba(fold_x_test)[:, 1]
            p_fold_train = self.model_catboost.predict_proba(fold_x_train)[:, 1]
            score = roc_auc_score(fold_y_test, p_fold)
            score_train = roc_auc_score(fold_y_train, p_fold_train)
            self.logger.info(f"Number of splits: {number_of_folds}; Fold: {i}; Test AUC: {round(score, 4)}; Train AUC: {round(score_train, 4)}")
            # Save in Level_1_train the "free" predictions concatenated
            train_df_level_1.loc[test_index.tolist(), 'train_y_hat'] = p_fold
            # Predict in test to make the k model mean
            # Define name of the prediction (p_"iteration number")
            if make_prediction:
                name = 'p_' + str(i)
                # Prediction to real test
                real_pred = self.model_catboost.predict_proba(x_test)[:, 1]
                # Name
                real_pred = pd.DataFrame({name: real_pred}, columns=[name])
                # Add to Level_1_test
                test_df_level_1 = pd.concat((test_df_level_1, real_pred), axis=1)
        # Compute the metric of the total concatenated prediction (and free of overfitting) in train
        score_total = roc_auc_score(y, train_df_level_1['train_y_hat'])
        self.logger.info(f"Number of splits: {number_of_folds}; Total AUC: {round((score_total) * 100, 4)} %")
        if make_prediction:
            test_df_level_1['model'] = test_df_level_1.mean(axis=1)
        # Return train and test sets with predictions and the performance
        if make_prediction:
            return train_df_level_1, pd.DataFrame({'test_y_hat': test_df_level_1['model']}), score_total
        else:
            return score_total

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
    parser.add_argument('-m', '--method', type=str, help='Learning method to use', required=True, choices=['random','catboost', 'catboost-kfold'])
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
    elif args.method == 'catboost-kfold':
        method = Catboost(logger, model_params={'eval_metric': 'AUC',
                                                'iterations': 5000,
                                                'od_type': 'Iter',
                                                'random_seed': RANDOM_SEED,
                                                'verbose': 50})
        train_df, test_df = method.process_data(train_df, test_df)
        x_train = train_df.drop(columns=['ID', 'TARGET', 'X24', 'X56', 'X39']).reset_index(drop=True)
        y_train = train_df['TARGET'].reset_index(drop=True)
        x_test = test_df.drop(columns=['X24', 'X56', 'X39']).reset_index(drop=True)
        method.train(x_train, y_train, run_params={'objective': 'Logloss',
                                                   'learning_rate': 0.01,  # learning rate, lower -> slower but better prediction
                                                   'depth': 6,  # Depth of the trees (values betwwen 5 and 10, higher -> more overfitting)
                                                   'min_data_in_leaf': 150,
                                                   'l2_leaf_reg': 20,  # L2 regularization (between 3 and 20, higher -> less overfitting)
                                                   'rsm': 0.5, # % of features to consider in each split (lower -> faster and reduces overfitting)
                                                   'subsample': 0.5,  # Sample rate for bagging
                                                   'random_seed': RANDOM_SEED
                                                   })
        number_of_rounds = round(method.model_catboost.best_iteration_ / (1 - TEST_TRAIN_RATIO) * (1 - 1 / NUMBER_OF_FOLDS))
        iterations = [round(number_of_rounds * f) for f in [0.9,1,1.1]]
        scores: List[float] = list()
        for i in iterations:
            logger.info(f'Iteration {i}')
            method = CatboostKFold(logger, model_params={'eval_metric': 'AUC',
                                                         'od_type': 'Iter',
                                                         'random_seed': 2305,
                                                         'n_estimators': i,
                                                         'verbose': False,}, random_seed=2305)
            Pred_train, Pred_test, score = method.train_predict(number_of_folds=NUMBER_OF_FOLDS,
                                                                random_seed=2305,
                                                                make_prediction=True, x=x_train, y=y_train,
                                                                x_test=x_test.drop(columns=['ID']).reset_index(drop=True),
                                                                run_params={'objective': 'Logloss',
                                                                   'learning_rate': 0.01,  # learning rate, lower -> slower but better prediction
                                                                   'depth': 6,  # Depth of the trees (values betwwen 5 and 10, higher -> more overfitting)
                                                                   'min_data_in_leaf': 150,
                                                                   'l2_leaf_reg': 20,  # L2 regularization (between 3 and 20, higher -> less overfitting)
                                                                   'rsm': 0.5, # % of features to consider in each split (lower -> faster and reduces overfitting)
                                                                   'subsample': 0.5,  # Sample rate for bagging
                                                                   'random_seed': 2305
                                                                })
            # Look if we are in the first test:
            if len(scores) == 0:
                max_score = float('-inf')
            else:
                max_score = max(scores)

            # If the score improves, we keep this one:
            if score >= max_score:
                print('BEST')
                Catboost_train = Pred_train.copy()
                Catboost_test = Pred_test.copy()

            # Append score
            scores.append(score)

        # The best cross-validated score has been found in:
        print('\n###########################################')
        print('Catboost optimal rounds: ', iterations[scores.index(max(scores))])
        print('Catboost optimal GINI: ', round((max(scores) * 2 - 1) * 100, 4), '%')
        print('Catboost optimal AUC: ', round(max(scores) * 100, 4), '%')
        print('###########################################')

        # 3) Train a model on whole train with the optimal parameters:
        ################################################################################

        # Adjust optimal CV number of rounds to whole sample size:
        i = int(iterations[scores.index(max(scores))] / (1 - 1 / NUMBER_OF_FOLDS))

        # Define the optimal model
        method = Catboost(logger, model_params={'n_estimators': i,
                                                'random_seed': 1234,
                                                'verbose': 100})
        x_train = train_df.drop(columns=['ID', 'TARGET', 'X24', 'X56', 'X39']).reset_index(drop=True)
        y_train = train_df['TARGET'].reset_index(drop=True)
        x_test = test_df.drop(columns=['X24', 'X56', 'X39']).reset_index(drop=True)
        method.train(x_train, y_train, run_params={'objective': 'Logloss',
                                                   'learning_rate': 0.01,  # learning rate, lower -> slower but better prediction
                                                   'depth': 4,  # Depth of the trees (values betwwen 5 and 10, higher -> more overfitting)
                                                   'min_data_in_leaf': 150,
                                                   'l2_leaf_reg': 20,  # L2 regularization (between 3 and 20, higher -> less overfitting)
                                                   'rsm': 0.5, # % of features to consider in each split (lower -> faster and reduces overfitting)
                                                   'subsample': 0.5,  # Sample rate for bagging
                                                   'random_seed': 1234
                                                   })
        result = method.predict(train_df.drop(columns=['TARGET', 'X24', 'X56', 'X39']).reset_index(drop=True))
        cm = confusion_matrix(y_train, result['Pred'] > 0.5)
        print(cm)
        result = method.predict(x_test)
    result.to_csv(args.result_file, index=False)
