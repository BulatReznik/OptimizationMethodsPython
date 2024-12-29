import warnings

import numpy as np
import time


class DepthFirstSearch:
    @staticmethod
    def dfs(graph, start, visited):
        # Использование явного стека для DFS
        stack = [start]

        while stack:
            node = stack.pop()

            if node not in visited:
                visited.add(node)
                # Добавляем всех непосещенных соседей в стек
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)


class LinearApproximator:
    @staticmethod
    def compute_coefficients(x, y):
        """
        Вычисляет коэффициенты линейной аппроксимации (a и b).
        """
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)

        # Вычисление коэффициентов
        a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        b = (sum_y - a * sum_x) / n
        return a, b

    def approximate(self, a, b, x):
        """
        Возвращает аппроксимированное значение y для заданного x.
        """
        return a * x + b


class FletcherReeves:
    @staticmethod
    def gradient(x):
        # Пример функции для минимизации: f(x, y) = (x-3)^2 + (y-2)^2
        return np.array([2 * (x[0] - 3), 2 * (x[1] - 2)])

    @staticmethod
    def minimize(max_iterations, epsilon, initial_guess):
        x = np.array(initial_guess)
        g = FletcherReeves.gradient(x)
        d = g.copy()

        for _ in range(max_iterations):
            grad_norm = np.linalg.norm(g)

            # Проверка на остановку по норме градиента
            if grad_norm < epsilon:
                break

            # Линейный шаг
            alpha = 0.1
            x = x - alpha * g

            # Новый градиент
            g_new = FletcherReeves.gradient(x)

            # Вычисление beta
            numerator = np.dot(g_new, g_new)
            denominator = np.dot(g, g)

            beta = numerator / denominator if denominator != 0 else 0

            # Обновление направления поиска
            d = g_new + beta * d
            g = g_new

        return x

class SimplexMethod:
    @staticmethod
    def simplex(type_, A, B, C, D, M):
        """
        Реализует симплекс-метод для решения задачи линейного программирования с использованием метода большого M.

        Аргументы:
        type_ -- тип оптимизации ('max' или 'min')
        A     -- матрица коэффициентов ограничений (numpy array)
        B     -- вектор свободных членов ограничений (numpy array)
        C     -- вектор коэффициентов целевой функции (numpy array)
        D     -- типы ограничений: 1 (<=), 0 (=), -1 (>=) (numpy array)
        M     -- большое значение M для метода большого M
        """
        # Количество ограничений и переменных
        m, n = A.shape
        basic_vars = []
        count = n
        R = np.eye(m)  # Единичная матрица для дополнительных переменных
        P = B  # Значения новых переменных
        artificial = []  # Искусственные переменные

        for i in range(m):
            if D[i] == 1:
                C = np.vstack((C, [[0]]))
                count += 1
                basic_vars.append(count - 1)
                artificial.append(0)
            elif D[i] == 0:
                C = np.vstack((C, [[M if type_ == 'min' else -M]]))
                count += 1
                basic_vars.append(count - 1)
                artificial.append(1)
            elif D[i] == -1:
                C = np.vstack((C, [[0], [M if type_ == 'min' else -M]]))
                R = SimplexMethod._repeat_column_negative(R, count + 1 - n)
                P = SimplexMethod._insert_zero_to_col(P, count + 1 - n)
                count += 2
                basic_vars.append(count - 1)
                artificial.append(0)
                artificial.append(1)

        X = np.vstack((np.zeros((n, 1)), P))
        A = np.hstack((A, R))
        st = np.vstack((np.hstack((-np.transpose(C), np.array([[0]]))), np.hstack((A, B))))


        # Итерации симплекс-метода
        while True:
            if type_ == 'min':
                w = np.amax(st[0, :-1])
                iw = np.argmax(st[0, :-1])
            else:
                w = np.amin(st[0, :-1])
                iw = np.argmin(st[0, :-1])

            if (type_ == 'min' and w <= 0) or (type_ == 'max' and w >= 0):
                break

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                T = st[1:, -1] / st[1:, iw]

            R = np.logical_and(T != np.inf, T > 0)
            min_value, ik = SimplexMethod._min_with_mask(T, R)
            pivot = st[ik + 1, iw]
            prow = st[ik + 1, :] / pivot
            st -= st[:, [iw]] * prow
            st[ik + 1, :] = prow
            basic_vars[ik] = iw

        z_optimal = st[0, -1]

        # Для вывода значений переменных x1 и x2
        x_values = np.zeros(n)
        for i in range(m):
            if basic_vars[i] < n:
                x_values[basic_vars[i]] = st[i + 1, -1]

        return z_optimal, x_values

    @staticmethod
    def _min_with_mask(x, mask):
        """Возвращает минимальное значение из x, отфильтрованное по маске."""
        min_value = float('inf')
        imin = -1
        for i, val in enumerate(x):
            if mask[i] and val < min_value:
                min_value = val
                imin = i
        return min_value, imin

    @staticmethod
    def _repeat_column_negative(mat, h):
        """Повторяет столбец h с отрицательными значениями."""
        r, c = mat.shape
        return np.hstack((mat[:, :h - 1], -mat[:, [h - 1]], mat[:, h - 1:]))

    @staticmethod
    def _insert_zero_to_col(col, h):
        """Добавляет нулевой элемент в колонку col на позицию h."""
        return np.vstack((col[:h - 1, [0]], [[0]], col[h - 1:, [0]]))

def lab3():
    # Генерация большого графа с 10,000 вершинами
    num_vertices = 10000
    edge_density = 0.05  # 5% возможных рёбер

    # Генерация случайного графа с использованием словаря списков
    import random
    graph = {i: [] for i in range(1, num_vertices + 1)}

    for i in range(1, num_vertices + 1):
        for j in range(i + 1, num_vertices + 1):
            if random.random() < edge_density:
                graph[i].append(j)
                graph[j].append(i)  # Граф неориентированный

    start_node = 1
    dfs = DepthFirstSearch()

    # Измеряем время для 100 запусков
    import time
    total_time = 0
    for _ in range(100):
        visited = set()
        start_time = time.time()
        dfs.dfs(graph, start_node, visited)
        total_time += time.time() - start_time

    # Вычисляем среднее время
    avg_time = total_time / 100

    # Один запуск для демонстрации
    visited_final = set()
    dfs.dfs(graph, start_node, visited_final)

    print(f"Обход графа завершен. Посещенные вершины: {', '.join(map(str, list(visited_final)[:10]))}...")
    print(f"\nСреднее время выполнения DFS (за 100 запусков): {avg_time:.6f} секунд")


def lab1():
    # Генерация больших данных
    size = 100000
    x = np.linspace(1, 1000, size)
    np.random.seed(42)  # Фиксируем генератор случайных чисел
    y = 2 * x + np.random.uniform(0, 0.5, size)

    if len(x) != len(y):
        print("Массивы x и y должны быть одной длины.")
        return

    approximator = LinearApproximator()

    # Суммируем время для 100 замеров
    total_time = 0
    for _ in range(100):
        start_time = time.time()
        a, b = approximator.compute_coefficients(x, y)
        elapsed_time = time.time() - start_time
        total_time += elapsed_time

    # Среднее время
    avg_time = total_time / 100

    # Выводим результаты
    print(f"Линейная аппроксимация: y = {a:.4f}x + {b:.4f}")
    print(f"\nСреднее время выполнения аппроксимации (за 100 замеров): {avg_time:.6f} секунд")

    print("\nПример аппроксимации первых 10 значений:")
    for xi, yi in zip(x[:10], y[:10]):
        y_approx = approximator.approximate(a, b, xi)
        print(f"x = {xi:.2f}, y (реальное) = {yi:.4f}, y (аппроксимация) = {y_approx:.4f}")


def lab2():
    # Исходные данные для минимизации с использованием Флетчера-Ривса
    initial_guess = [10, 10]  # Начальная точка для минимизации
    max_iterations = 1000
    epsilon = 1e-6

    start_time = time.time()

    # Вызов метода минимизации Флетчера-Ривса
    result = FletcherReeves.minimize(max_iterations, epsilon, initial_guess)

    elapsed_time = time.time() - start_time

    print(f"\nРезультат минимизации с использованием Флетчера-Ривса: {result}")
    print(f"Время выполнения алгоритма Флетчера-Ривса: {elapsed_time:.6f} секунд")


def lab4():
    # Задача линейного программирования: найти максимум функции z = 3x + 5y
    # при ограничениях:
    # x + 2y <= 8
    # 2x + y <= 10
    # x, y >= 0

    A = np.array([[2, 1], [-2, 3], [2, 4]])  # Матрица коэффициентов ограничений
    B = np.array([[10], [6], [-8]])  # Вектор правых частей ограничений
    C = np.array([[2], [3]])  # Вектор коэффициентов целевой функции
    D = np.array([1, 1, -1])  # Типы ограничений (1 - <=, -1 - >=)
    M = 1000  # Большое значение для метода большого M

    simplex = SimplexMethod()

    # Замер времени выполнения симплекс-метода
    total_time = 0
    for _ in range(100):
        start_time = time.time()
        result, x_values = simplex.simplex('max', A, B, C, D, M)
        total_time += time.time() - start_time

    avg_time = total_time / 100
    print(f"\nРезультат симплекс-метода: {result:.4f}")
    print(f"Среднее время выполнения (за 100 запусков): {avg_time:.6f} секунд")

    # Вывод значений переменных x1 и x2
    print(f"Значения переменных: x1 = {x_values[0]:.4f}, x2 = {x_values[1]:.4f}")

if __name__ == "__main__":
    # lab1()
    # lab2()
    # lab3()
    lab4()
