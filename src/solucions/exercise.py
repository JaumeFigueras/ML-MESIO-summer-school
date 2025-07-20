#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import pandas as pd
import numpy as np

from logging.handlers import RotatingFileHandler
from sklearn.metrics import confusion_matrix

from src.methods.random import Random
from src.methods.catboost_basic import CatboostBasic
from src.methods.catboost_cross_validation import CatboostCrossValidation

from typing import Optional
from typing import Dict
from typing import Any
from typing import List
from typing import Iterable
from typing import Union


def remove_missing_values(dfs: Union[Iterable[pd.DataFrame], pd.DataFrame], threshold: int) -> Union[List[pd.DataFrame], pd.DataFrame]:
    """
    Removes the dataframe rows with more than threshold missing values

    :param dfs: List of data frames or dataframe to be processed
    :type dfs: Iterable[pd.DataFrame] or pd.DataFrame
    :param threshold: Number of missing values to consider when removing the row
    :type threshold: int
    :return: List of processed data frames
    :rtype: Iterable[pd.DataFrame] or pd.DataFrame
    """

    outputs: List[pd.DataFrame] = list()
    dfs_to_process = dfs if isinstance(dfs, list) else [dfs]
    for df in dfs_to_process:
        # Remove rows with more than threshold missing values
        ids_with_many_nans = df.loc[df.isna().sum(axis=1) >= threshold, 'ID']
        df = df[df.isna().sum(axis=1) < threshold].copy()
        # Reindex
        df.reset_index(drop=True, inplace=True)
        # Add the data frame to a processed list
        outputs.append(df)
    return outputs if len(outputs) > 1 else outputs[0]

def fill_missing_categorical(dfs: Union[Iterable[pd.DataFrame], pd.DataFrame], fill_value: str) -> Union[List[pd.DataFrame], pd.DataFrame]:
    """
    Fill missing categorical columns with a fill value

    :param dfs: List of data frames or dataframe to be processed
    :type dfs: Iterable[pd.DataFrame] or pd.DataFrame
    :param fill_value: Value to fill missing values with
    :type fill_value: str
    :return: List of processed data frames
    :rtype: Iterable[pd.DataFrame] or pd.DataFrame
    """
    outputs: List[pd.DataFrame] = list()
    dfs_to_process = dfs if isinstance(dfs, Iterable) else [dfs]
    for df in dfs_to_process:
        # Fill NaN in categorical columns with 'NA'
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        df[cat_cols] = df[cat_cols].copy().fillna('NA')
        outputs.append(df)
    return outputs if len(outputs) > 1 else outputs[0]

if __name__ == "__main__":
    """ Main program to run the different ML approaches
    Should be called like:
    python -m src.solucions.exercise --train-file ./data/train.csv --test-file ./data/test.csv --result-file ./results/random.csv --method random
    """
    # Config the program arguments
    # noinspection DuplicatedCode
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train-file', type=str, help='Train CSV file', required=True)
    parser.add_argument('-s', '--test-file', type=str, help='Test CSV file', required=True)
    parser.add_argument('-r', '--result-file', type=str, help='Result CSV file to write', required=True)
    parser.add_argument('-m', '--method', type=str, help='Learning method to use', required=True, choices=['random','catboost', 'catboost-cv', 'prof'])
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

    result: pd.DataFrame = pd.DataFrame()
    if args.method == 'random':
        # Constant definition for the basic CatBoost
        THRESHOLD_MISSING: int = 20
        # Pre process data
        train_df, test_df = remove_missing_values((train_df, test_df), THRESHOLD_MISSING)
        train_df, test_df = fill_missing_categorical((train_df, test_df), 'NA')
        # Fill a result with random values
        method = Random(logger)
        method.train(train_df, pd.DataFrame())
        result = method.predict(test_df, 'ID', 'Pred')
    if args.method == 'catboost':
        # Constant definition for the basic CatBoost
        THRESHOLD_MISSING: int = 20
        THRESHOLD_CORRELATION: float = 0.99
        TEST_TRAIN_RATIO: float = 0.3
        RANDOM_SEED: int = 1234
        EARLY_STOPPING_ROUNDS: int = 100
        # Parameters of the CatBoost model
        model_parameters = {'eval_metric': 'AUC',
                            'iterations': 5000,
                            'od_type': 'Iter',
                            'random_seed': RANDOM_SEED,
                            'verbose': 50
                            }
        run_parameters = {'objective': 'Logloss',
                          'learning_rate': 0.01,  # learning rate, lower -> slower but better prediction
                          'depth': 4,  # Depth of the trees (values between 5 and 10, higher -> more overfitting)
                          'min_data_in_leaf': 150,
                          'l2_leaf_reg': 20,  # L2 regularization (between 3 and 20, higher -> less overfitting)
                          'rsm': 0.5,  # % of features to consider in each split (lower -> faster and reduces overfitting)
                          'subsample': 0.5,  # Sample rate for bagging
                          'random_seed': RANDOM_SEED
                          }
        other_parameters = {'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
                            'random_seed': RANDOM_SEED,
                            'test_train_ratio': TEST_TRAIN_RATIO
                           }
        # Pre process data
        train_df = remove_missing_values(train_df, THRESHOLD_MISSING)
        train_df, test_df = fill_missing_categorical((train_df, test_df), 'NA')
        # CatBoost Basic
        method = CatboostBasic(logger, model_params=model_parameters)
        # Process data
        logger.info("Processing data")
        logger.info(f"Processing Correlation")
        correlated_columns = method.analyse_correlation(train_df, THRESHOLD_CORRELATION)
        if correlated_columns is not None and len(correlated_columns) > 0:
            train_df = train_df.drop(columns=correlated_columns)
            test_df = test_df.drop(columns=correlated_columns)
        logger.info(f"Processing Constant columns")
        constant_columns = method.analyse_constant(train_df)
        if constant_columns is not None and len(constant_columns) > 0:
            train_df = train_df.drop(columns=constant_columns)
            test_df = test_df.drop(columns=constant_columns)
        x_train = train_df.drop(columns=['ID', 'TARGET']).reset_index(drop=True)
        y_train = train_df['TARGET'].reset_index(drop=True)
        x_test = test_df.reset_index(drop=True)
        # Train and predict
        method.train(x_train, y_train, run_params=run_parameters, other_params=other_parameters)
        result = method.predict(x_test, 'ID', 'Pred')
    elif args.method == 'catboost-cv':
        less_significant: List[str] = [ 'X59', 'X57', 'X47', 'X31', 'X37']
        # less_significant: List[str] = []
        root_train_df = train_df.drop(columns=less_significant).copy()
        root_test_df = test_df.drop(columns=less_significant).copy()
        # OTHER
        DEPTH = 7
        MIN_DATA_LEAF = 200
        THRESHOLD_VALUES = 20
        # Basic CatBoost run to determine the number of iterations that was used to learn
        # Constant definition for the basic CatBoost
        THRESHOLD_MISSING: int = THRESHOLD_VALUES
        THRESHOLD_CORRELATION: float = 0.99
        TEST_TRAIN_RATIO: float = 0.3
        RANDOM_SEED: int = 1234
        EARLY_STOPPING_ROUNDS: int = 100
        # Parameters of the CatBoost model
        model_parameters = {'eval_metric': 'AUC',
                            'iterations': 5000,
                            'od_type': 'Iter',
                            'random_seed': RANDOM_SEED,
                            'verbose': 50
                            }
        run_parameters = {'objective': 'Logloss',
                          'learning_rate': 0.01,  # learning rate, lower -> slower but better prediction
                          'depth': DEPTH,  # Depth of the trees (values between 5 and 10, higher -> more overfitting)
                          'min_data_in_leaf': MIN_DATA_LEAF,
                          'l2_leaf_reg': 20,  # L2 regularization (between 3 and 20, higher -> less overfitting)
                          'rsm': 0.5,  # % of features to consider in each split (lower -> faster and reduces overfitting)
                          'subsample': 0.5,  # Sample rate for bagging
                          'random_seed': RANDOM_SEED
                          }
        other_parameters = {'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
                            'random_seed': RANDOM_SEED,
                            'test_train_ratio': TEST_TRAIN_RATIO
                           }
        # Pre process data
        train_df = root_train_df.copy()
        test_df = root_test_df.copy()
        train_df = remove_missing_values(train_df, THRESHOLD_MISSING)
        train_df, test_df = fill_missing_categorical((train_df, test_df), 'NA')
        # CatBoost Basic
        method = CatboostBasic(logger, model_params=model_parameters)
        # Process data
        logger.info("Processing data")
        logger.info(f"Processing Correlation")
        correlated_columns = method.analyse_correlation(train_df, THRESHOLD_CORRELATION)
        if correlated_columns is not None and len(correlated_columns) > 0:
            train_df = train_df.drop(columns=correlated_columns)
            test_df = test_df.drop(columns=correlated_columns)
        logger.info(f"Processing Constant columns")
        constant_columns = method.analyse_constant(train_df)
        if constant_columns is not None and len(constant_columns) > 0:
            train_df = train_df.drop(columns=constant_columns)
            test_df = test_df.drop(columns=constant_columns)
        x_train = train_df.drop(columns=['ID', 'TARGET']).reset_index(drop=True)
        y_train = train_df['TARGET'].reset_index(drop=True)
        x_test = test_df.reset_index(drop=True)
        # Train and predict
        method.train(x_train, y_train, run_params=run_parameters, other_params=other_parameters)
        # Cross validation CatBoost
        # First restore the original data
        train_df = root_train_df.copy()
        test_df = root_test_df.copy()
        # Constant definition for the basic CatBoost
        THRESHOLD_MISSING: int = 20
        THRESHOLD_CORRELATION: float = 0.99
        TEST_TRAIN_RATIO: float = 0.3
        RANDOM_SEED: int = 2305
        EARLY_STOPPING_ROUNDS: int = 100
        NUMBER_OF_FOLDS: int = 5
        # ITERATIONS_MULTIPLIER_LIST: List[float] = [0.9, 1, 1.1]
        ITERATIONS_MULTIPLIER_LIST: List[float] = [0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3]
        # Parameters of the CatBoost model
        model_parameters = {'eval_metric': 'AUC',
                            'od_type': 'Iter',
                            'random_seed': RANDOM_SEED,
                            'verbose': False
                            }
        run_parameters = {'objective': 'Logloss',
                          'learning_rate': 0.01,  # learning rate, lower -> slower but better prediction
                          'depth': DEPTH,  # Depth of the trees (values between 5 and 10, higher -> more overfitting)
                          'min_data_in_leaf': MIN_DATA_LEAF,
                          'l2_leaf_reg': 20,  # L2 regularization (between 3 and 20, higher -> less overfitting)
                          'rsm': 0.5,  # % of features to consider in each split (lower -> faster and reduces overfitting)
                          'subsample': 0.5,  # Sample rate for bagging
                          'random_seed': RANDOM_SEED
                          }
        other_parameters = {'number_of_folds': NUMBER_OF_FOLDS, 'random_seed': RANDOM_SEED}
        number_of_iterations = round(method.model_catboost.best_iteration_ / (1 - TEST_TRAIN_RATIO) * (1 - 1 / NUMBER_OF_FOLDS))
        iterations = [round(number_of_iterations * f) for f in ITERATIONS_MULTIPLIER_LIST]
        scores: List[float] = list()
        # Process data
        logger.info("Processing data")
        train_df = remove_missing_values(train_df, THRESHOLD_MISSING)
        train_df, test_df = fill_missing_categorical((train_df, test_df), 'NA')
        logger.info(f"Processing Correlation")
        correlated_columns = method.analyse_correlation(train_df, THRESHOLD_CORRELATION)
        if correlated_columns is not None and len(correlated_columns) > 0:
            train_df = train_df.drop(columns=correlated_columns)
            test_df = test_df.drop(columns=correlated_columns)
        logger.info(f"Processing Constant columns")
        constant_columns = method.analyse_constant(train_df)
        if constant_columns is not None and len(constant_columns) > 0:
            train_df = train_df.drop(columns=constant_columns)
            test_df = test_df.drop(columns=constant_columns)
        x_train = train_df.drop(columns=['ID', 'TARGET']).reset_index(drop=True)
        y_train = train_df['TARGET'].reset_index(drop=True)
        x_test = test_df.reset_index(drop=True)
        for i in iterations:
            logger.info(f'Iteration {i}')
            method = CatboostCrossValidation(logger, model_params={**model_parameters,
                                                                   'n_estimators': i
                                                                   })

            Pred_train, Pred_test, score = method.train_predict(x=x_train, y=y_train,
                                                                x_test=x_test.drop(columns=['ID']).reset_index(drop=True),
                                                                run_params=run_parameters, other_params= other_parameters)
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
        # Basic CatBoost run to determine the number of iterations that was used to learn
        # Constant definition for the basic CatBoost
        THRESHOLD_MISSING: int = 20
        THRESHOLD_CORRELATION: float = 0.99
        TEST_TRAIN_RATIO: float = 0.3
        RANDOM_SEED: int = 1234
        EARLY_STOPPING_ROUNDS: int = 10000
        # Parameters of the CatBoost model
        model_parameters = {'eval_metric': 'AUC',
                            'n_estimators': i,
                            'od_type': 'Iter',
                            'random_seed': RANDOM_SEED,
                            'verbose': 50
                            }
        run_parameters = {'objective': 'Logloss',
                          'learning_rate': 0.01,  # learning rate, lower -> slower but better prediction
                          'depth': DEPTH,  # Depth of the trees (values between 5 and 10, higher -> more overfitting)
                          'min_data_in_leaf': MIN_DATA_LEAF,
                          'l2_leaf_reg': 20,  # L2 regularization (between 3 and 20, higher -> less overfitting)
                          'rsm': 0.5,  # % of features to consider in each split (lower -> faster and reduces overfitting)
                          'subsample': 0.5,  # Sample rate for bagging
                          'random_seed': RANDOM_SEED
                          }
        other_parameters = {'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
                            'random_seed': RANDOM_SEED,
                            'test_train_ratio': 0.0
                           }
        # Pre process data
        train_df = remove_missing_values(train_df, THRESHOLD_MISSING)
        train_df, test_df = fill_missing_categorical((train_df, test_df), 'NA')
        # CatBoost Basic
        method = CatboostBasic(logger, model_params=model_parameters)
        # Process data
        logger.info("Processing data")
        logger.info(f"Processing Correlation")
        correlated_columns = method.analyse_correlation(train_df, THRESHOLD_CORRELATION)
        if correlated_columns is not None and len(correlated_columns) > 0:
            train_df = train_df.drop(columns=correlated_columns)
            test_df = test_df.drop(columns=correlated_columns)
        logger.info(f"Processing Constant columns")
        constant_columns = method.analyse_constant(train_df)
        if constant_columns is not None and len(constant_columns) > 0:
            train_df = train_df.drop(columns=constant_columns)
            test_df = test_df.drop(columns=constant_columns)
        x_train = train_df.drop(columns=['ID', 'TARGET']).reset_index(drop=True)
        y_train = train_df['TARGET'].reset_index(drop=True)
        x_test = test_df.reset_index(drop=True)
        # Train and predict
        method.train(x_train, y_train, run_params=run_parameters, other_params=other_parameters)

        result = method.predict(train_df.drop(columns=['TARGET']).reset_index(drop=True), 'ID', 'Pred')
        cm = confusion_matrix(y_train, result['Pred'] > 0.5)
        print(cm)
        result = method.predict(x_test, 'ID', 'Pred')

    elif args.method == 'prof':
        ################################################################################
        ################################# FEATURE ######################################
        ############################### ENGINEERING ####################################
        ################################################################################
        # Feature types
        train = train_df.copy()
        test = test_df.copy()
        Features = train.dtypes.reset_index()
        Categorical = Features.loc[Features[0] == 'object', 'index']

        # Categorical to the begining
        cols = train.columns.tolist()
        pos = 0
        for col in Categorical:
            cols.insert(pos, cols.pop(cols.index(col)))
            pos += 1
        train = train[cols]
        cols.remove('TARGET')
        test = test[cols]


        # 1) Missings
        ################################################################################
        # Function to print columns with at least n_miss missings
        def miss(ds, n_miss):
            miss_list = list()
            for col in list(ds):
                if ds[col].isna().sum() >= n_miss:
                    print(col, ds[col].isna().sum())
                    miss_list.append(col)
            return miss_list


        # Which columns have 1 missing at least...
        print('\n################## TRAIN ##################')
        m_tr = miss(train, 1)
        print('\n################## TEST ##################')
        m_te = miss(test, 1)

        # Missings in categorical features (fix it with an 'NA' string)
        ################################################################################
        train, test = fill_missing_categorical((train, test), 'NA')
        # for col in Categorical:
        #     train.loc[train[col].isna(), col] = 'NA'
        #     test.loc[test[col].isna(), col] = 'NA'

        # Missings -> Drop some rows
        ################################################################################
        # We can see a lot of colummns with 3 missings in train, look the data and...
        # there are 4 observations that have many columns with missing values:
        # A1039
        # A2983
        # A3055
        # A4665
        train = remove_missing_values(train, 20)

        train.reset_index(drop=True, inplace=True)

        # 2) Correlations
        ################################################################################
        # Let's see if certain columns are correlated
        # or even that are the same with a "shift"
        thresholdCorrelation = 0.99


        def InspectCorrelated(df):
            corrMatrix = df.corr().abs()  # Correlation Matrix
            upperMatrix = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k=1).astype(bool))
            correlColumns = []
            for col in upperMatrix.columns:
                correls = upperMatrix.loc[upperMatrix[col] > thresholdCorrelation, col].keys()
                if len(correls) >= 1:
                    correlColumns.append(col)
                    print("\n", col, '->', end=" ")
                    for i in correls:
                        print(i, end=" ")
            print('\nSelected columns to drop:\n', correlColumns)
            return correlColumns


        # Look at correlations in the original features
        correlColumns = InspectCorrelated(train.iloc[:, len(Categorical):-1])

        # If we are ok, throw them:
        train = train.drop(correlColumns, axis=1)
        test = test.drop(correlColumns, axis=1)


        # 3) Constants
        ################################################################################
        # Let's see if there is some constant column:
        def InspectConstant(df):
            consColumns = []
            for col in list(df):
                if len(df[col].unique()) < 2:
                    print(df[col].dtypes, '\t', col, len(df[col].unique()))
                    consColumns.append(col)
            print('\nSelected columns to drop:\n', consColumns)
            return consColumns


        consColumns = InspectConstant(train.iloc[:, len(Categorical):-1])

        # If we are ok, throw them:
        train = train.drop(consColumns, axis=1)
        test = test.drop(consColumns, axis=1)

        train['TARGET'].mean()

        ################################################################################
        ################################ MODEL CATBOOST ################################
        ################################# TRAIN / TEST #################################
        ################################################################################
        pred = list(train)[1:-1]
        X_train = train[pred].reset_index(drop=True)
        Y_train = train['TARGET'].reset_index(drop=True)
        X_test = test[pred].reset_index(drop=True)

        # 1) For expensive models (catboost) we first try with validation set (no cv)
        ################################################################################
        from catboost import CatBoostClassifier
        from catboost import Pool

        # train / test partition
        RS = 1234  # Seed for partitions (train/test) and model random part
        TS = 0.3  # Validation size
        esr = 100  # Early stopping rounds (when validation does not improve in these rounds, stops)

        from sklearn.model_selection import train_test_split

        x_tr, x_val, y_tr, y_val = train_test_split(X_train, Y_train, test_size=TS, random_state=RS)

        # Categorical positions for catboost
        Pos = list()
        As_Categorical = Categorical.tolist()
        print(As_Categorical)
        As_Categorical.remove('ID')
        for col in As_Categorical:
            Pos.append((X_train.columns.get_loc(col)))

        # To Pool Class (for catboost only)
        pool_tr = Pool(x_tr, y_tr, cat_features=Pos)
        pool_val = Pool(x_val, y_val, cat_features=Pos)

        # By-hand parameter tuning. A grid-search is expensive
        # We test different combinations
        # See parameter options here:
        # "https://catboost.ai/en/docs/references/training-parameters/"
        model_catboost_val = CatBoostClassifier(
            eval_metric='AUC',
            iterations=5000,  # Very high value, to find the optimum
            od_type='Iter',  # Overfitting detector set to "iterations" or number of trees
            random_seed=RS,  # Random seed for reproducibility
            verbose=50)  # Shows train/test metric every "verbose" trees

        # "Technical" parameters of the model:
        params = {'objective': 'Logloss',
                  'learning_rate': 0.01,  # learning rate, lower -> slower but better prediction
                  'depth': 4,  # Depth of the trees (values betwwen 5 and 10, higher -> more overfitting)
                  'min_data_in_leaf': 150,
                  'l2_leaf_reg': 20,  # L2 regularization (between 3 and 20, higher -> less overfitting)
                  'rsm': 0.5,  # % of features to consider in each split (lower -> faster and reduces overfitting)
                  'subsample': 0.5,  # Sample rate for bagging
                  'random_seed': RS}

        model_catboost_val.set_params(**params)

        print('\nCatboost Fit (Validation)...\n')
        model_catboost_val.fit(X=pool_tr,
                               eval_set=pool_val,
                               early_stopping_rounds=esr)

    result.to_csv(args.result_file, index=False)
