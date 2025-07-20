#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from catboost import Pool
from sklearn.model_selection import train_test_split

from src.methods.method import Method

from typing import Optional
from typing import Dict
from typing import Any
from typing import List
from typing import Union
from logging import Logger

class CatboostBasic(Method):
    def __init__(self, l: Logger, model_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Default constructor

        :param l: Logger instance
        :type l: Logger
        :param model_params: Model parameters
        :type model_params: Dict[str, Any]
        """
        super().__init__(l, model_params)
        self.model_catboost = CatBoostClassifier(**self.params)

    def train(self, x: pd.DataFrame, y: pd.DataFrame, run_params: Optional[Dict[str, Any]] = None, other_params: Optional[Dict[str, Any]] = None) -> None:
        self.logger.info("Catboost training started")
        self.logger.info(f"Catboost training parameters: {self.params}")
        ttr = 0.3
        if other_params is not None and 'test_train_ratio' in other_params:
            ttr = other_params['test_train_ratio']
        rs = 1234
        if other_params is not None and 'random_seed' in other_params:
            rs = other_params['random_seed']
        esr = 100
        if other_params is not None and 'early_stopping_rounds' in other_params:
            esr = other_params['early_stopping_rounds']
        categorical: List[str] = list(x.select_dtypes(include=['object', 'category']).columns)
        if ttr > 0.0:
            x_to_train, x_to_validate, y_to_train, y_to_validate = train_test_split(x, y, test_size=ttr, random_state=rs)
        else:
            x_to_train = x
            y_to_train = y
        pool_tr = Pool(x_to_train, y_to_train, cat_features=categorical)
        if ttr > 0.0:
            pool_val = Pool(x_to_validate, y_to_validate, cat_features=categorical)
        self.model_catboost.set_params(**run_params)
        if ttr > 0.0:
            self.model_catboost.fit(X=pool_tr, eval_set=pool_val, early_stopping_rounds=esr)
        else:
            self.model_catboost.fit(X=pool_tr, early_stopping_rounds=esr)
        self.logger.info("Catboost training ended")

    def predict(self, x: pd.DataFrame, extra_column_names: Union[List[str], str, None], result_column_name: str) -> pd.DataFrame:
        self.logger.info("Catboost prediction started")
        y = super().predict(x, extra_column_names, result_column_name)
        y[result_column_name] = self.model_catboost.predict_proba(x.drop(columns=extra_column_names))[:, 1]
        self.logger.info("Catboost prediction ended")
        return y
