#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from abc import ABC
from abc import abstractmethod

from typing import Optional
from typing import Dict
from typing import Any
from typing import List
from typing import Union
from logging import Logger


class Method(ABC):
    """
    Abstract class to manage all learning methods tested
    """

    def __init__(self, l: Logger, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Default constructor

        :param l: Logger to write verbose messages
        :type l: Logger
        :param params: Parameters to pass to the learning method
        :type params: Dict[str, Any]
        """
        self.params = params if params is not None else dict()
        self.logger = l

    @abstractmethod
    def train(self, x: pd.DataFrame, y:pd.DataFrame, run_params: Optional[Dict[str, Any]] = None, other_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Abstract train function. Normally it should do some precessing and call the method fit procedure

        :param x: The train X data frame
        :type x: pd.DataFrame
        :param y: The train Y data frame
        :type y: pd.DataFrame
        :param run_params: The specific parameters to pass to the learning method in the fit procedure
        :type run_params: Dict[str, Any]
        :return: Nothing
        :rtype: None
        """
        pass

    @staticmethod
    def predict(x: pd.DataFrame, extra_column_names: Union[List[str], str, None], result_column_name: str) -> pd.DataFrame:
        """
        Default prediction function, just prepares the output data frame with the existing colum names from X and a mew
        column with the result

        :param x: The X data set to predict
        :type x: pd.DataFrame
        :param extra_column_names: List of column names to copy to the prediction data frame
        :type extra_column_names: List[str]
        :param result_column_name: Name of the prediction result column
        :type result_column_name: str
        :return: The blank prediction data frame
        :rtype: pd.DataFrame
        """
        y = pd.DataFrame()
        if extra_column_names is not None:
            if isinstance(extra_column_names, str):
                y = x[[extra_column_names]].copy()
            elif isinstance(extra_column_names, list):
                y = x[extra_column_names].copy()
        y[result_column_name] = 0
        return y

    def analyse_correlation(self, df: pd.DataFrame, correlation_threshold: float) -> Union[List[str], None]:
        """
        Analyzes the correlation between all the columns of the data frame and returns a list of column manes with
        correlation greater or equal to the threshold

        :param df: Data frame to analyze
        :type df: pd.DataFrame
        :param correlation_threshold: Threshold to be considered as correlated columns
        :type correlation_threshold: float
        :return: The list of correlated column names that are equal or over the threshold
        :rtype: List[str] or None
        """
        # get the Pearson correlation matrix
        numerical_df = df.select_dtypes(include=['number'])
        correlation_matrix = numerical_df.corr().abs()  # Correlation Matrix
        # Extract the upper triangular, excluding the diagonal and marking True to the elements that we like to
        # analyze and NaN to those wee don't like to analyze. Where is equivalent to a & mask
        upper_matrix = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        correlated_columns: List[str] = [col for col in upper_matrix.columns if
                                         any(upper_matrix[col] > correlation_threshold)]
        self.logger.info(f"Correlated columns: {', '.join(correlated_columns) if len(correlated_columns) else 'None'}")
        return correlated_columns if len(correlated_columns) > 0 else None

    def analyse_constant(self, df: pd.DataFrame) -> Union[List[str], None]:
        """
        Analyzes the remote possibility that two columns have the same values and return a list of duplicated column
        names
        :param df: Data frame to analyze
        :type df: pd.DataFrame
        :return: The list of column names that are an exact copy
        :rtype: List[str] or None
        """
        constant_columns =  [col for col in df.columns if df[col].nunique() < 2]
        self.logger.info(f"Constant columns: {', '.join(constant_columns) if len(constant_columns) > 0 else 'None'}")
        return constant_columns if len(constant_columns) > 0 else None

    @staticmethod
    def remove_missing(df: pd.DataFrame, missing_threshold: int) -> pd.DataFrame:
        return df[df.isna().sum(axis=1) < missing_threshold].copy()