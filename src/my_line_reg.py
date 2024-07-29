
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class MyLineReg:
    ''' Реализация класса линейной регрессии

    Параметры:
    learning_rate (float, optional): Скорость обучения (по умолчанию 0.01).
    n_iter (int, optional): Количество итераций обучения (по умолчанию 1000).
    metric (str, optional): Метрика для отслеживания (по умолчанию None).
    reg (str, optional): Тип регуляризации (по умолчанию None).
    l1_coef (float, optional): Коэффициент L1 регуляризации (по умолчанию 0.0).
    l2_coef (float, optional): Коэффициент L2 регуляризации (по умолчанию 0.0).
    sgd_sample (int, optional): Размер выборки для SGD (по умолчанию None).
    random_state (int, optional): Генератор случайных чисел (по умолчанию 42).

    Методы:
    fit(X, y, verbose=False): Обучение модели.
    get_coef(): Получение весов модели.
    predict(X): Предсказание.
    get_best_score(): Получение лучшего значения метрики.
    _calculate_metric(y, y_hat): Расчет значения метрики.

    Атрибуты:
    weights: Веса модели.
    best_score: Лучшее значение метрики.
    metric: Метрика для отслеживания.
    learning_rate: Скорость обучения.
    n_iter: Количество итераций обучения.
    reg: Тип регуляризации.
    l1_coef: Коэффициент L1 регуляризации.
    l2_coef: Коэффициент L2 регуляризации.
    sgd_sample: Размер выборки для SGD.
    random_state: Генератор случайных чисел.

    '''

    def __init__(self, learning_rate=0.01, n_iter=1000, metric=None, reg=None, l1_coef=0.0, l2_coef=0.0, sgd_sample=None, random_state=42):
        self.learning_rate = learning_rate  # Скорость обучения
        self.n_iter = n_iter  # Количество итераций
        self.weights = None  # Веса модели
        self.metric = metric  # Метрика для отслеживания
        self.best_score = None  # Лучшее значение метрики
        self.reg = reg  # Тип регуляризации
        self.l1_coef = l1_coef  # Коэффициент L1 регуляризации
        self.l2_coef = l2_coef  # Коэффициент L2 регуляризации
        self.sgd_sample = sgd_sample  # Размер выборки для SGD
        self.random_state = random_state  # Генератор случайных чисел

    def fit(self, X, y, verbose=False):
        X = np.c_[np.ones(X.shape[0]), X]  # Добавляем столбец единиц слева
        self.weights = np.ones(X.shape[1])  # Инициализируем веса единицами

        for i in range(1, self.n_iter + 1):
            y_hat = np.dot(X, self.weights)  # Предсказание
            loss = np.mean((y_hat - y) ** 2)  # Среднеквадратичная ошибка (MSE)

            # Добавление регуляризации к лоссу
            if self.reg == 'l1':
                loss += self.l1_coef * np.sum(np.abs(self.weights))
            elif self.reg == 'l2':
                loss += self.l2_coef * np.sum(self.weights ** 2)
            elif self.reg == 'elasticnet':
                loss += self.l1_coef * np.sum(np.abs(self.weights)) + self.l2_coef * np.sum(self.weights ** 2)

            # Обновление весов
            gradient = 2 * np.dot(X.T, (y_hat - y)) / X.shape[0]

            # Добавление регуляризации к градиенту
            if self.reg == 'l1':
                gradient += self.l1_coef * np.sign(self.weights)
            elif self.reg == 'l2':
                gradient += 2 * self.l2_coef * self.weights
            elif self.reg == 'elasticnet':
                gradient += self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights

            # Определение скорости обучения
            if callable(self.learning_rate):
                lr = self.learning_rate(i)
            else:
                lr = self.learning_rate

            self.weights -= lr * gradient  # Обновление весов

            if verbose and i % verbose == 0:
                metric_value = self._calculate_metric(y, y_hat)
                if self.metric:
                    print(f"{i} | loss: {loss:.2f} | {self.metric}: {metric_value:.2f} | lr: {lr:.6f}")
                else:
                    print(f"{i} | loss: {loss:.2f} | lr: {lr:.6f}")

        # Сохранение последнего значения метрики после завершения обучения
        y_hat = np.dot(X, self.weights)
        self.best_score = self._calculate_metric(y, y_hat)

        if verbose:
            if self.metric:
                print(f"Final | loss: {loss:.2f} | {self.metric}: {self.best_score:.2f}")
            else:
                print(f"Final | loss: {loss:.2f}")

    def get_coef(self):
        return self.weights[1:]  # Возвращаем веса (кроме первого элемента)

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]  # Добавляем столбец единиц слева
        y_pred = np.dot(X, self.weights)  # Предсказание
        return y_pred  # Возвращаем вектор предсказаний

    def get_best_score(self):
        return self.best_score  # Возвращаем последнее значение метрики

    def _calculate_metric(self, y, y_hat):
        if self.metric == 'mae':
            return mean_absolute_error(y, y_hat)
        elif self.metric == 'mse':
            return mean_squared_error(y, y_hat)
        elif self.metric == 'rmse':
            return np.sqrt(mean_squared_error(y, y_hat))
        elif self.metric == 'mape':
            return np.mean(np.abs((y - y_hat) / y)) * 100
        elif self.metric == 'r2':
            return r2_score(y, y_hat)
        else:
            return None
