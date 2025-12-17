import numpy as np
import pandas as pd
from collections import Counter


def calculate_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNNClassifier:
    """
    Реализация метода k-ближайших соседей для классификации
    """

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, x, y):
        """Сохранение обучающих данных"""
        # Преобразуем в numpy массивы для единообразия
        self.X_train = np.array(x)
        self.y_train = np.array(y)
        return self

    def predict(self, x_test):
        """Предсказание классов для новых данных"""
        x_test = np.array(x_test)
        y_predicted = np.zeros(x_test.shape[0], dtype=self.y_train.dtype)

        # Для каждой точки из тестового набора
        for i, test_point in enumerate(x_test):
            # 1. Вычисляем расстояния до всех обучающих точек
            distances = []
            for j, train_point in enumerate(self.X_train):
                dist = calculate_distance(test_point, train_point)
                distances.append((dist, self.y_train[j]))

            # 2. Сортируем по расстоянию (от ближайшего к дальнему)
            distances.sort(key=lambda x: x[0])

            # 3. Выбираем k ближайших соседей
            k_nearest = distances[:self.k]

            # 4. Извлекаем метки классов соседей
            k_nearest_labels = [label for _, label in k_nearest]

            # 5. Выбираем наиболее частый класс (режим)
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            y_predicted[i] = most_common
        return y_predicted
