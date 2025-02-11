import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CustomStockScaler(BaseEstimator, TransformerMixin):
    def __init__(self, y_min=0.8, y_max=1.2):
        """
        :param y_min: float, минимальное значение для y после скейлинга
        :param y_max: float, максимальное значение для y после скейлинга
        """
        self.y_min = y_min
        self.y_max = y_max
        self.feature_mean = None
        self.feature_std = None
        self.feature_columns = None  # Будет задано в fit()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Обучает скейлер: вычисляет среднее и стандартное отклонение для X.
        :param X: pd.DataFrame, фичи (кроме y)
        :param y: pd.Series, целевая переменная
        """
        self.feature_columns = X.columns  # Запоминаем, какие фичи были переданы
        self.feature_mean = X.mean()
        self.feature_std = X.std()
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Применяет масштабирование к X и y.
        :param X: pd.DataFrame, фичи
        :param y: pd.Series, целевая переменная
        :return: (X_scaled, y_scaled) - кортеж с преобразованными данными
        """
        X_scaled = (X - self.feature_mean) / self.feature_std
        y_scaled = self._scale_y(y)
        return X_scaled, y_scaled

    def inverse_transform(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        Обратное преобразование X и y.
        :param X: pd.DataFrame, масштабированные фичи
        :param y: pd.Series, масштабированное значение y
        :return: (X_original, y_original) - кортеж с восстановленными данными
        """
        X_unscaled = X * self.feature_std + self.feature_mean
        y_unscaled = self._inverse_scale_y(y)
        return X_unscaled, y_unscaled

    def _scale_y(self, y: pd.Series) -> pd.Series:
        """Сжимает y в диапазон [y_min, y_max]"""
        center = (self.y_max + self.y_min) / 2
        scale = (self.y_max - self.y_min) / 2
        return center + scale * np.tanh((y - 1) * 2)

    def _inverse_scale_y(self, y_scaled: pd.Series) -> pd.Series:
        """Обратное преобразование y"""
        center = (self.y_max + self.y_min) / 2
        scale = (self.y_max - self.y_min) / 2
        return 1 + (np.arctanh((y_scaled - center) / scale) / 2)