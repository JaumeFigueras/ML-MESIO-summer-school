#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from src.methods.method import Method

from typing import Optional
from typing import Dict
from typing import Any
from typing import List
from typing import Union
from logging import Logger


class Random(Method):
    """
    Dummy class that predict random numbers. It is mainly used for testing purposes.
    """
    def __init__(self, l: Logger, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Default constructor.

        :param l: Logger to write verbose messages
        :type l: Logger
        :param params: Parameters to pass to the learning method
        :type params: Dict[str, Any]
        """
        super().__init__(l, params)

    def train(self, x: pd.DataFrame, y: pd.DataFrame, run_params: Optional[Dict[str, Any]] = None, other_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Dummy train function. Normally it should do some precessing and call the method fit procedure

        :param x: The train X data frame
        :type x: pd.DataFrame
        :param y: The train Y data frame
        :type y: pd.DataFrame
        :param run_params: The specific parameters to pass to the learning method in the fit procedure
        :type run_params: Dict[str, Any]
        :return: Nothing
        :rtype: None
        """
        self.logger.info("Random training started")
        self.logger.info(f"Random training parameters: {self.params}")
        self.logger.info("Random training ended")

    def predict(self, x: pd.DataFrame, extra_column_names: Union[List[str], str, None], result_column_name: str) -> pd.DataFrame:
        """
        Random prediction function. Fills the prediction values with random numbers [0, 1)

        :param x: The X data set to predict
        :type x: pd.DataFrame
        :param extra_column_names: List of column names to copy to the prediction data frame
        :type extra_column_names: List[str]
        :param result_column_name: Name of the prediction result column
        :type result_column_name: str
        :return: The blank prediction data frame
        :rtype: pd.DataFrame
        """
        self.logger.info("Random prediction started")
        y = super().predict(x, extra_column_names, result_column_name)
        y[result_column_name] = np.random.rand(len(y))
        self.logger.info("Random prediction ended")
        return y
