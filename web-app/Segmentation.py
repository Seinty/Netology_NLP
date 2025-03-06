import tensorflow as tf
from typing import Dict


class SentimentModel(tf.keras.Model):
    def __init__(self, input_dim: int, **kwargs) -> None:
        """
        Инициализирует модель с заданной размерностью входных данных.

        Параметры:
        - input_dim (int): Размерность входных данных.
        - **kwargs: Дополнительные параметры, передаваемые в конструктор базового класса.

        Конструктор создает два полносвязных слоя:
        - fc1: Полносвязный слой с 128 нейронами и функцией активации ReLU.
        - fc2: Полносвязный слой с 1 нейроном и функцией активации Sigmoid.
        """
        super(SentimentModel, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.fc1 = tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,))
        self.fc2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(
            self,
            x: tf.Tensor
    ) -> tf.Tensor:
        """
        Выполняет прямой проход данных через модель.

        Параметры:
        - x (tf.Tensor): Входные данные, которые будут пропущены через модель.

        Возвращаемое значение:
        - tf.Tensor: Результаты работы модели после применения двух полносвязных слоев.
        """
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def get_config(
            self
    ) -> Dict[str, int]:
        """
        Возвращает конфигурацию модели в виде словаря.

        Возвращаемое значение:
        - dict: Словарь с конфигурацией модели, включая размерность входных данных.
        """
        config = super(SentimentModel, self).get_config()
        config.update({"input_dim": self.input_dim})
        return config

    @classmethod
    def from_config(
            cls,
            config: Dict[str, int]
    ) -> 'SentimentModel':
        """
        Создает экземпляр модели из предоставленной конфигурации.

        Параметры:
        - config (dict): Словарь с конфигурацией модели.

        Возвращаемое значение:
        - SentimentModel: Новый экземпляр модели, инициализированный параметрами из конфигурации.
        """
        return cls(**config)
