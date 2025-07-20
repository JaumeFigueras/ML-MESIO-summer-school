#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from catboost import Pool

from src.methods.catboost_basic import CatboostBasic

from typing import Optional
from typing import Dict
from typing import Any
from typing import List
from typing import Union
from logging import Logger


class CatboostCrossValidation(CatboostBasic):
    def __init__(self, l: Logger, model_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Default constructor, sends all params to the parent

        :param l: Logger instance
        :type l: Logger
        :param model_params: Model parameters
        :type model_params: Dict[str, Any]
        """
        super().__init__(l, model_params)

    def train_predict(self, x: pd.DataFrame, y: pd.DataFrame, x_test: pd.DataFrame, run_params: Dict[str, Any], other_params: Dict[str, Any]) -> Any:
        nof = 5
        if other_params is not None and 'number_of_folds' in other_params:
            nof = other_params['number_of_folds']
        rs = 1234
        if other_params is not None and 'random_seed' in other_params:
            rs = other_params['random_seed']
        # Create the k folds
        k_folds = StratifiedKFold(n_splits=nof, shuffle=True, random_state=rs)
        train_df_level_1 = pd.DataFrame(np.zeros((x.shape[0], 1)), columns=['train_y_hat'])
        test_df_level_1 = pd.DataFrame()
        self.model_catboost.set_params(**run_params)
        for i, (train_index, test_index) in enumerate(k_folds.split(x, y)):
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
            prediction_fold_test = self.model_catboost.predict_proba(fold_x_test)[:, 1]
            prediction_fold_train = self.model_catboost.predict_proba(fold_x_train)[:, 1]
            score_test = roc_auc_score(fold_y_test, prediction_fold_test)
            score_train = roc_auc_score(fold_y_train, prediction_fold_train)
            self.logger.info(f"Fold: {i} of {nof}; Test AUC: {round(score_test, 4)}; Train AUC: {round(score_train, 4)}")
            # Save in Level_1_train the "free" predictions concatenated
            train_df_level_1.loc[test_index.tolist(), 'train_y_hat'] = prediction_fold_test
            # Predict in test to make the k model mean
            # Define name of the prediction (p_"iteration number")
            name = 'p_' + str(i)
            # Prediction to real test
            real_pred = self.model_catboost.predict_proba(x_test)[:, 1]
            # Name
            real_pred = pd.DataFrame({name: real_pred}, columns=[name])
            # Add to Level_1_test
            test_df_level_1 = pd.concat((test_df_level_1, real_pred), axis=1)
        # Compute the metric of the total concatenated prediction (and free of overfitting) in train
        score_total = roc_auc_score(y, train_df_level_1['train_y_hat'])
        self.logger.info(f"Total AUC: {round((score_total) * 100, 4)} %")
        test_df_level_1['model'] = test_df_level_1.mean(axis=1)
        # Return train and test sets with predictions and the performance
        return train_df_level_1, pd.DataFrame({'test_y_hat': test_df_level_1['model']}), score_total

