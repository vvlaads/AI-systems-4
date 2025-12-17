import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import get_resource_path
from knn import KNNClassifier


def normalize_features(x_data):
    """Нормализация признаков"""
    mean = np.mean(x_data, axis=0)
    std = np.std(x_data, axis=0)

    # Избегаем деления на 0
    std[std == 0] = 1
    return (x_data - mean) / std


def train_test_split_custom(X, y, test_size=0.2, random_state=42):
    """Разделение на обучающую и тестовую выборки"""
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    test_count = int(len(X) * test_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def cm3(y_true, y_pred):
    """Матрица ошибок для 3 классов"""
    cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for t, p in zip(y_true, y_pred):
        cm[t - 1][p - 1] += 1  # Классы 1,2,3 -> индексы 0,1,2
    return cm


def plot_cm(ax, cm, k, accuracy):
    """Рисует одну матрицу ошибок"""
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=np.max(cm) * 1.2)
    ax.set_title(f'k={k} ({accuracy:.2%})', fontsize=10)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['1', '2', '3'])
    ax.set_yticklabels(['1', '2', '3'])

    # Добавляем значения
    for i in range(3):
        for j in range(3):
            color = 'white' if cm[i][j] > np.max(cm) / 2 else 'black'
            ax.text(j, i, str(cm[i][j]), ha='center', va='center', color=color, fontsize=12)


# Загрузка CSV
pd.set_option('display.max_columns', None)
df = pd.read_csv(get_resource_path("WineDataset.csv"))

df = df.dropna()  # Убрать пустые строки

print("=" * 100)
print("Основная статистика:")
print(df.describe())

# Гистограммы
df.hist(bins=30,  # Количество интервалов
        figsize=(14, 8),  # Размер графика (ширина, высота)
        edgecolor="black",  # Цвет границ столбцов
        layout=(4, 4))  # Расположение графиков (строки, столбцы)
plt.suptitle('Гистограммы распределения признаков', fontsize=14)
plt.tight_layout()

# Box plot
df.plot(kind='box', subplots=True, layout=(4, 4), figsize=(14, 8),
        showfliers=False,  # Не отображать выбросы
        showmeans=True,  # Показывать среднее значение
        meanline=True,  # Отображать среднее значение линией
        meanprops={'linestyle': '--', 'linewidth': 2, 'color': 'red'})
plt.suptitle('Box-plots', fontsize=14)
plt.tight_layout()

# 3D-визуализация с цветами по классам
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Цветовая схема для классов вин
colors = {1: 'red', 2: 'green', 3: 'blue'}
class_names = {1: 'Класс 1', 2: 'Класс 2', 3: 'Класс 3'}

# Разделяем данные по классам и отображаем разными цветами
for wine_class in sorted(df['Wine'].unique()):
    class_data = df[df['Wine'] == wine_class]
    ax.scatter(class_data['Alcohol'],
               class_data['Malic Acid'],
               class_data['Ash'],
               s=40, alpha=0.7,
               color=colors[wine_class],
               label=class_names[wine_class])

# Настройка графика
ax.set_xlabel('Alcohol', fontsize=12, labelpad=10)
ax.set_ylabel('Malic Acid', fontsize=12, labelpad=10)
ax.set_zlabel('Ash', fontsize=12, labelpad=10)
ax.set_title('3D-визуализация: Классы вин по химическому составу', fontsize=14, pad=15)
ax.legend(title='Тип вина')
ax.grid(True, alpha=0.3)
plt.tight_layout()

# Разделяем признаки и целевую переменную
target = 'Wine'
X = df.drop(target, axis=1).values
y = df[target].values

# Нормализация данных
X_normalized = normalize_features(X)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split_custom(
    X_normalized, y, test_size=0.2, random_state=42
)

print("=" * 100)
print("Размеры выборок:")
print(f"  Обучающая: {X_train.shape[0]} образцов")
print(f"  Тестовая:  {X_test.shape[0]} образцов")
print("=" * 100)

k_values = [1, 3, 5, 7, 10]

# Модель 1: случайные
random.seed(42)
rand_feat = random.sample([c for c in df.columns if c != 'Wine'], 3)
X1 = df[rand_feat].values
X1_norm = (X1 - X1.mean(0)) / (X1.std(0) + 1e-10)
X1_train, X1_test = train_test_split_custom(X1_norm, y, test_size=0.2)[:2]
print("МОДЕЛЬ 1 (случайные признаки):")

# Матрицы для модели 1
cms1 = []
accs1 = []
for k in k_values:
    knn = KNNClassifier(k=k)
    knn.fit(X1_train, y_train)
    y_pred = knn.predict(X1_test)
    acc = np.mean(y_pred == y_test)
    cm = cm3(y_test, y_pred)

    cms1.append(cm)
    accs1.append(acc)

    print(f"k={k}: {acc:.2%} | Матрица: {cm}")

print("=" * 100)

# Модель 2: фиксированные
fix_feat = ['Alcohol', 'Malic Acid', 'Ash']
X2 = df[fix_feat].values
X2_norm = (X2 - X2.mean(0)) / (X2.std(0) + 1e-10)
X2_train, X2_test = train_test_split_custom(X2_norm, y, test_size=0.2)[:2]
print("МОДЕЛЬ 2 (фиксированные признаки):")

# Матрицы для модели 2
cms2 = []
accs2 = []
for k in k_values:
    knn = KNNClassifier(k=k)
    knn.fit(X2_train, y_train)
    y_pred = knn.predict(X2_test)
    acc = np.mean(y_pred == y_test)
    cm = cm3(y_test, y_pred)

    cms2.append(cm)
    accs2.append(acc)

    print(f"k={k}: {acc: .2%} | Матрица: {cm}")

# Создаем фигуры
fig1, axes1 = plt.subplots(1, len(k_values), figsize=(15, 3))
fig2, axes2 = plt.subplots(1, len(k_values), figsize=(15, 3))

# Модель 1: все матрицы в одной строке
for i, (k, cm, acc) in enumerate(zip(k_values, cms1, accs1)):
    plot_cm(axes1[i] if len(k_values) > 1 else axes1, cm, k, acc)
    if i == 0:
        axes1[i].set_ylabel('Истинный класс', fontsize=10)
    axes1[i].set_xlabel('Предсказанный', fontsize=10)

fig1.suptitle('Модель 1: Матрицы ошибок для разных k', fontsize=14)

# Модель 2: все матрицы в одной строке
for i, (k, cm, acc) in enumerate(zip(k_values, cms2, accs2)):
    plot_cm(axes2[i] if len(k_values) > 1 else axes2, cm, k, acc)
    if i == 0:
        axes2[i].set_ylabel('Истинный класс', fontsize=10)
    axes2[i].set_xlabel('Предсказанный', fontsize=10)

fig2.suptitle('Модель 2: Матрицы ошибок для разных k', fontsize=14)

plt.tight_layout()
plt.show()
