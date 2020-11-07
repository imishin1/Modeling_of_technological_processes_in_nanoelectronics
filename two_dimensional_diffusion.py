import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

from one_dimensional_diffusion import Physical_quantities, Initial_param, Culculation_param, List_of_param

class Matrix_of_param():
    def __init__(self, n, m):
        self.zero_matrix = np.zeros((n, m))

class Draw_graph3D(Culculation_param):
    def draw3D(self):
        n = 50
        m = 50
        dt = 10

        # введем размеры образца и окна в сантиметрах
        x_window = 1E-4
        xmax = 3 / 10000
        ymax = 3 / 10000

        # расчитаем приращение каждого шага
        dx = xmax / (n - 1)
        dy = ymax / (m - 1)

        # создадим необходимые нулевые списки и матрицы
        zero_matrix = Matrix_of_param(n, m).zero_matrix
        C_matrix = Matrix_of_param(n, m).zero_matrix
        C2_matrix = Matrix_of_param(n, m).zero_matrix     
        x_list = List_of_param(n).new_list
        y_list = List_of_param(m).new_list
        a_list = List_of_param(n).new_list
        b_list = List_of_param(n).new_list
        d_list = List_of_param(n).new_list
        delta_list = List_of_param(n).new_list
        lambda_list = List_of_param(n).new_list
        r_list = List_of_param(n).new_list
        p_n = List_of_param(n).new_list

        # заполним списки x и y с заданным шагом
        for i in range(0, n):
            x_list[i] = dx * i
        for i in range(0, m):
            y_list[i] = dy * i
        
        # введем начальные условия для примеси в окне размером 1 мкм
        for j in range (0, m):
            if (y_list[j] > ((ymax - x_window) / 2)) and (y_list[j] < ((ymax - x_window) / 2 + x_window)):
                C_matrix[0, j] = self.C0

        for i in range (1, n-1):
            b_list[i] = 1
            d_list[i] = 1  

        # загонка
        for k in range (0, self.t_zagonka,  dt):
            # X
            for j in range (0, m-1):
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

                for i in range (1, n - 2):
                    a_list[i] = - (2 + ((dx ** 2) / (self.find_Diff(C_matrix[i, j], self.T_zagonka) * dt)))
                    r_list[i] = - (dx ** 2) / (dy ** 2) * (C_matrix[i+1, j] - (2 - (dy ** 2) / self.find_Diff(C_matrix[i, j], self.T_zagonka) / dt) *\
                    C_matrix[i, j] + C_matrix[i-1,j])
                    delta_list[i] = - d_list[i] / (a_list[i] + b_list[i] * delta_list[i - 1])
                    lambda_list[i] = (r_list[i] - b_list[i] * lambda_list[i - 1]) / (a_list[i] + b_list[i] * delta_list[i - 1])
                C_matrix[n-1, j] = lambda_list[n-1]
                for i in range(n - 2, -1, -1):
                    C_matrix[i, j] = delta_list[i] * C_matrix[i + 1, j] + lambda_list[i]

            b_list[0] = 0
            a_list[0] = -1
            d_list[0] = 1
            r_list[0] = 0
            delta_list[0] = -d_list[0] / a_list[0]
            lambda_list[0] = r_list[0] / a_list[0]
            # Y
            for i in range (0, n-1):
                for j in range (1, m - 2):
                    a_list[j] = - (2 + (dy ** 2) / (self.find_Diff(C_matrix[i, j], self.T_zagonka) * dt))
                    r_list[j] = - (dy ** 2) / (dx ** 2) * (C_matrix[i, j+1] - (2 - (dx ** 2) / self.find_Diff(C_matrix[i, j], self.T_zagonka) / dt) *\
                    C_matrix[i, j] + C_matrix[i, j-1])
                    delta_list[j] = - d_list[j] / (a_list[j] + b_list[j] * delta_list[j - 1])
                    lambda_list[j] = (r_list[j] - b_list[j] * lambda_list[j - 1]) / (a_list[j] + b_list[j] * delta_list[j - 1])
                C_matrix[i, m-1] = lambda_list[n-1]
                for j in range(m - 2, -1, -1):
                    C_matrix[i, j] = delta_list[j] * C_matrix[i, j + 1] + lambda_list[j]

        xo_list, yo_list = np.meshgrid(x_list, y_list)

        fig = plt.figure()
        axes = fig.add_subplot(projection='3d')

        axes.plot_wireframe(yo_list, xo_list, C_matrix, color='#FF8C00')

        # разгонка
        b_list[0] = 0
        a_list[0] = -1
        d_list[0] = 1
        r_list[0] = 0
        delta_list[0] = -d_list[0] / a_list[0]
        lambda_list[0] = r_list[0] / a_list[0]        
        for k in range (0, self.t_razgonka,  dt):
            # X
            for j in range (0, m-1):
                for i in range (1, n - 2):
                    a_list[i] = - (2 + ((dx ** 2) / (self.find_Diff(C_matrix[i, j], self.T_razgonka) * dt)))
                    r_list[i] = - (dx ** 2) / (dy ** 2) * (C_matrix[i+1, j] - (2 - (dy ** 2) / self.find_Diff(C_matrix[i, j], self.T_razgonka) / dt) *\
                    C_matrix[i, j] + C_matrix[i-1,j])
                    delta_list[i] = - d_list[i] / (a_list[i] + b_list[i] * delta_list[i - 1])
                    lambda_list[i] = (r_list[i] - b_list[i] * lambda_list[i - 1]) / (a_list[i] + b_list[i] * delta_list[i - 1])
                C_matrix[n-1, j] = lambda_list[n-1]
                for i in range(n - 2, -1, -1):
                    C_matrix[i, j] = delta_list[i] * C_matrix[i + 1, j] + lambda_list[i]
            # Y
            for i in range (0, n-1):
                for j in range (1, m - 2):
                    a_list[j] = - (2 + (dy ** 2) / (self.find_Diff(C_matrix[i, j], self.T_razgonka) * dt))
                    r_list[j] = - (dy ** 2) / (dx ** 2) * (C_matrix[i, j+1] - (2 - (dx ** 2) / self.find_Diff(C_matrix[i, j], self.T_razgonka) / dt) *\
                    C_matrix[i, j] + C_matrix[i, j-1])
                    delta_list[j] = - d_list[j] / (a_list[j] + b_list[j] * delta_list[j - 1])
                    lambda_list[j] = (r_list[j] - b_list[j] * lambda_list[j - 1]) / (a_list[j] + b_list[j] * delta_list[j - 1])
                C_matrix[i, m-1] = lambda_list[n-1]
                for j in range(m - 2, -1, -1):
                    C_matrix[i, j] = delta_list[j] * C_matrix[i, j + 1] + lambda_list[j]

        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.set_zlabel('z')

        axes.plot_wireframe(yo_list, xo_list, C_matrix)

        axes.set_title('Двумерная загонка и разгонка')

        plt.show()

        # собираем список с модулями разницы C_As - C_list[i], чтобы найти минимальное значение и определить глубину залегания p-n перехода
        for i in range(0, n):
            p_n[i] = np.absolute(self.C_As - C_matrix[i, int(n / 2)])

        # возвращает индекс наименьшего элемента массива p_n
        min_p_n = np.argmin(p_n)
        print(f'глубина залегания p-n перехода => {round(x_list[min_p_n] * 10000, 3)} мкм')

def main():
    # объявляем объем и запускаем функции рисования
    As = Draw_graph3D()
    As.draw3D()

# запускаем тело программы
if __name__ == "__main__":
    main()