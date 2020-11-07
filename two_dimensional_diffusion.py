# Импортируем необходимые библиотеки для расчетов и отрисовки
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

# Импортируем необходимые классы из расчетов одномерной диффузии
from one_dimensional_diffusion import Physical_quantities, Initial_param, Culculation_param, List_of_param

# Создадим класс для объявления нулевой матрицы размером n и m
class Matrix_of_param():
    def __init__(self, n, m):
        self.zero_matrix = np.zeros((n, m))

# Создадим класс для расчетов и отрисовки дмумерной диффузии
class Draw_graph3D(Culculation_param):
    def draw3D(self):
        n = 50
        m = 50
        dt = 10

        # Введем размеры образца и окна в сантиметрах
        x_window = 1E-4
        xmax = 3 / 10000
        ymax = 3 / 10000

        # Расчитаем приращение каждого шага
        dx = xmax / (n - 1)
        dy = ymax / (m - 1)

        # Объявим необходимые нулевые списки и матрицы
        zero_matrix = Matrix_of_param(n, m).zero_matrix
        C_matrix = Matrix_of_param(n, m).zero_matrix 
        x_list = List_of_param(n).new_list
        y_list = List_of_param(m).new_list
        a_list = List_of_param(n).new_list
        b_list = List_of_param(n).new_list
        d_list = List_of_param(n).new_list
        delta_list = List_of_param(n).new_list
        lambda_list = List_of_param(n).new_list
        r_list = List_of_param(n).new_list
        p_n = List_of_param(n).new_list

        # Заполним списки x и y с заданным шагом
        for i in range(0, n):
            x_list[i] = dx * i
        for i in range(0, m):
            y_list[i] = dy * i
        
        # Введем начальные условия для примеси в окне с заданным размером
        for j in range (0, m):
            if (y_list[j] > ((ymax - x_window) / 2)) and (y_list[j] < ((ymax - x_window) / 2 + x_window)):
                C_matrix[0, j] = self.C0

        # Задаем значения коэффициентов b и d для всех случаев, кроме граничных условий
        for i in range (1, n-1):
            b_list[i] = 1
            d_list[i] = 1  

        # Задаем граничные условия для коэффициентов при n
        b_list[n-1] = 0
        d_list[n-1] = 0
        r_list[n-1] = 0
        a_list[n-1] = 1

        # Загонка
        for k in range (0, self.t_zagonka,  dt):
            # По X
            for j in range (0, m-1):
                # Задаем граничные условия при нулевом индекесе для окна и за окном
                if (y_list[j] > ((ymax - x_window) / 2)) and (y_list[j] < ((ymax - x_window) / 2 + x_window)):
                    b_list[0] = 0
                    a_list[0] = 1
                    d_list[0] = 0
                    r_list[0] = self.C0
                else:
                    b_list[0] = 0
                    a_list[0] = -1
                    d_list[0] = 1
                    r_list[0] = 0
                delta_list[0] = -d_list[0] / a_list[0]
                lambda_list[0] = r_list[0] / a_list[0]
                
                # Рассчитываем прогоночные коэффициенты
                for i in range (1, n - 2):
                    a_list[i] = - (2 + ((dx ** 2) / (self.find_Diff(C_matrix[i, j], self.T_zagonka) * dt)))
                    r_list[i] = - (dx ** 2) / (dy ** 2) * (C_matrix[i+1, j] - (2 - (dy ** 2) / self.find_Diff(C_matrix[i, j], self.T_zagonka) / dt) *\
                    C_matrix[i, j] + C_matrix[i-1,j])
                    delta_list[i] = - d_list[i] / (a_list[i] + b_list[i] * delta_list[i - 1])
                    lambda_list[i] = (r_list[i] - b_list[i] * lambda_list[i - 1]) / (a_list[i] + b_list[i] * delta_list[i - 1])

                C_matrix[n-1, j] = lambda_list[n-1]

                # Заполняем матрицу концентраций
                for i in range(n - 2, -1, -1):
                    C_matrix[i, j] = delta_list[i] * C_matrix[i + 1, j] + lambda_list[i]

            # По Y
            # Задаем граничные условия при нулевом индексе
            b_list[0] = 0
            a_list[0] = -1
            d_list[0] = 1
            r_list[0] = 0
            delta_list[0] = -d_list[0] / a_list[0]
            lambda_list[0] = r_list[0] / a_list[0]

            for i in range (0, n-1):
                # Рассчитываем прогоночные коэффициенты
                for j in range (1, m - 2):
                    a_list[j] = - (2 + (dy ** 2) / (self.find_Diff(C_matrix[i, j], self.T_zagonka) * dt))
                    r_list[j] = - (dy ** 2) / (dx ** 2) * (C_matrix[i, j+1] - (2 - (dx ** 2) / self.find_Diff(C_matrix[i, j], self.T_zagonka) / dt) *\
                    C_matrix[i, j] + C_matrix[i, j-1])
                    delta_list[j] = - d_list[j] / (a_list[j] + b_list[j] * delta_list[j - 1])
                    lambda_list[j] = (r_list[j] - b_list[j] * lambda_list[j - 1]) / (a_list[j] + b_list[j] * delta_list[j - 1])

                C_matrix[i, m-1] = lambda_list[n-1]
                
                # Заполняем матрицу концентраций
                for j in range(m - 2, -1, -1):
                    C_matrix[i, j] = delta_list[j] * C_matrix[i, j + 1] + lambda_list[j]
        
        # Создаем координатную сетку из списков x и y
        xo_list, yo_list = np.meshgrid(x_list, y_list)

        # Объявляем форму для отрисовки и создаем оси
        fig = plt.figure()
        axes = fig.add_subplot(projection='3d')

        # Отрисовываем загонку
        axes.plot_wireframe(yo_list, xo_list, C_matrix, color='#FF8C00')

        # Разгонка
        # Задаем граничные условия при нулевом индексе
        b_list[0] = 0
        a_list[0] = -1
        d_list[0] = 1
        r_list[0] = 0
        delta_list[0] = -d_list[0] / a_list[0]
        lambda_list[0] = r_list[0] / a_list[0]

        for k in range (0, self.t_razgonka,  dt):
            # По X
            for j in range (0, m-1):
                # Рассчитываем прогоночные коэффициенты
                for i in range (1, n - 2):
                    a_list[i] = - (2 + ((dx ** 2) / (self.find_Diff(C_matrix[i, j], self.T_razgonka) * dt)))
                    r_list[i] = - (dx ** 2) / (dy ** 2) * (C_matrix[i+1, j] - (2 - (dy ** 2) / self.find_Diff(C_matrix[i, j], self.T_razgonka) / dt) *\
                    C_matrix[i, j] + C_matrix[i-1,j])
                    delta_list[i] = - d_list[i] / (a_list[i] + b_list[i] * delta_list[i - 1])
                    lambda_list[i] = (r_list[i] - b_list[i] * lambda_list[i - 1]) / (a_list[i] + b_list[i] * delta_list[i - 1])

                C_matrix[n-1, j] = lambda_list[n-1]
                
                # Заполняем матрицу концентраций
                for i in range(n - 2, -1, -1):
                    C_matrix[i, j] = delta_list[i] * C_matrix[i + 1, j] + lambda_list[i]

            # По Y
            for i in range (0, n-1):
                # Рассчитываем прогоночные коэффициенты
                for j in range (1, m - 2):
                    a_list[j] = - (2 + (dy ** 2) / (self.find_Diff(C_matrix[i, j], self.T_razgonka) * dt))
                    r_list[j] = - (dy ** 2) / (dx ** 2) * (C_matrix[i, j+1] - (2 - (dx ** 2) / self.find_Diff(C_matrix[i, j], self.T_razgonka) / dt) *\
                    C_matrix[i, j] + C_matrix[i, j-1])
                    delta_list[j] = - d_list[j] / (a_list[j] + b_list[j] * delta_list[j - 1])
                    lambda_list[j] = (r_list[j] - b_list[j] * lambda_list[j - 1]) / (a_list[j] + b_list[j] * delta_list[j - 1])

                C_matrix[i, m-1] = lambda_list[n-1]
                
                # Заполняем матрицу концентраций
                for j in range(m - 2, -1, -1):
                    C_matrix[i, j] = delta_list[j] * C_matrix[i, j + 1] + lambda_list[j]

        # Отрисовываем разгонку 
        axes.plot_wireframe(yo_list, xo_list, C_matrix)
        
        # Настраиваем названия осей и заголовка
        axes.set_title('Двумерная загонка и разгонка')
        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_zlabel('z')

        # Вызываем функцию демонстрации формы
        plt.show()

        # Собираем список с модулями разницы C_As - C_matrix[i, int(n / 2)], чтобы найти минимальное значение и определить глубину залегания p-n перехода
        for i in range(0, n):
            p_n[i] = np.absolute(self.C_As - C_matrix[i, int(n / 2)])

        # Возвращаем индекс наименьшего элемента массива p_n
        min_p_n = np.argmin(p_n)

        # Выводим на экран глубину залегания p-n перехода при индексе наименьшего элемента массива p-n
        print(f'глубина залегания p-n перехода => {round(x_list[min_p_n] * 10000, 3)} мкм')

def main():
    # Объявляем экземпляр класса и запускаем функцию рисования
    As = Draw_graph3D()
    As.draw3D()

# запускаем тело программы
if __name__ == "__main__":
    main()