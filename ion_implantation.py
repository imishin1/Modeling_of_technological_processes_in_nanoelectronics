# Импорт основных физических параметров и функции задания массивов из предыдущих расчетных заданий
from one_dimensional_diffusion import Physical_quantities, List_of_param
# Импорт необходимых библиотек для расчетов и построений зависимости
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Mishin Ilya
# The prosecc of ion implantation
# Si
# As
# E = 200 keV
# Q = 1E15 sm^-3

# Класс, содержащий основные параметры для расчетов
class Implantation_param():
    def __init__(self):
        self.E = 200 * 1000 # eV
        self.Q = 1E15 # sm^-3
        self.a0_Bor = 0.529 # A
        self.q2 = 14.4 # eV * A
        self.M_Si = 28.086
        self.Z_Si = 14
        self.M_As = 74.92
        self.Z_As = 33
        self.p_Si = 5E22 # sm^-3
        # self.x = float(input()) * 0.0001 # Можно сделать, чтоб глубина задавалась более гибко, при этом ввод с клавиатуры осуществляется в мкм
        self.x = 0.5 * 0.0001 # sm
    
class Culculation_param:
    def __init__(self):
        Physical_quantities.__init__(self)
        Implantation_param.__init__(self)
    
    # Расчет постоянной экранирования
    def a_ecran(self):
        return 0.8854 * self.a0_Bor / (((self.Z_Si ** (2 / 3)) + (self.Z_As ** (2 / 3))) ** 0.5) # A
    
    def eps(self, E):
        return (self.a_ecran() * self.M_As * E) / (self.Z_As * self.Z_Si * self.q2 * (self.M_Si + self.M_As))

    # Расчет ядерной тормозной способности
    def Sn_eps(self, E):
        a = 1.1383
        b = 0.01321
        c = 0.21226
        d = 0.19593
        if self.eps(E) > 10:
            return np.log(self.eps(E)) / (2 * self.eps(E))
        else:
            return np.log(1 + a * self.eps(E)) / (2 * (self.eps(E) + (b * self.eps(E) ** c) + (d * self.eps(E) ** 0.5)))
    
    # Перерасчет тормозной способности
    def Sn(self, E):
        return (self.p_Si * 8.462E-15 * self.Z_Si * self.Z_As * self.M_Si * self.Sn_eps(E)) / ((self.M_Si + self.M_As) * ((self.M_Si ** 0.23) + (self.M_As ** 0.23)))

    # Расчет электронной тормозной способности
    def Se(self, E):
        # Расчет коэффициента Линдхарда
        k = self.Z_Si ** (1 / 6) * 0.0793 * self.Z_Si ** 0.5 * (self.M_Si + self.M_As) ** 1.5 / ((self.Z_As ** (2 / 3) + self.Z_Si ** (2 / 3)) ** (3 / 4) * \
        (self.M_Si ** 1.5 * self.M_As ** 0.5))
        Cr = (4 * 3.14 * ((self.a_ecran() * 0.00000001) ** 2) * self.M_Si * self.M_As) / ((self.M_Si + self.M_As) ** 2)
        Ce = (self.a_ecran() * self.M_As) / (self.Z_Si * self.Z_As * self.q2 * (self.M_Si + self.M_As))
        # Расчет коэффициента пропорциональности
        K = k * (Cr) / (Ce ** 0.5)
        return K * (E ** 0.5) * self.p_Si

class Draw_graph(Culculation_param):
    def draw(self):
        n = 2000
        # Расчет приращений по шагу
        dE = self.E / n
        dx = self.x / n
        # Задание всех необходимых массиво (изначально с нулевыми значениями)
        Rp_list = List_of_param(n).new_list
        dRp_list = List_of_param(n).new_list
        koef_ksi_list = List_of_param(n).new_list
        xm_list = List_of_param(n).new_list
        xd_list = List_of_param(n).new_list
        C_list = List_of_param(n).new_list
        x_list = List_of_param(n).new_list

        for i in range (1, n):
            E = i * dE
            # Расчеты пробегов ионов в твердых телах
            Rp_list[i] = Rp_list[i - 1] * (1 - (self.M_As * self.Sn(E) * dE) / (2 * self.M_Si * (self.Sn(E) + self.Se(E)) * E)) + dE / (self.Sn(E) + self.Se(E))
            # Расчет разброса проецированного пробега dRp
            dRp_list[i] = ((dRp_list[i - 1] ** 2) + (koef_ksi_list[i - 1] - 2 * (dRp_list[i - 1] ** 2)) * (self.M_As * self.Sn(E) * dE) / \
            (self.M_Si * (self.Sn(E) + self.Se(E)) * E)) ** 0.5
            # Расчет коэффициента кси
            koef_ksi_list[i] = koef_ksi_list[i - 1] + (2 * Rp_list[i] * dE) / (self.Sn(E) + self.Se(E))
            
        dRp = ((2 / 3) * (Rp_list[n - 1] ** 2) * self.M_Si * self.M_As / ((self.M_Si + self.M_As) ** 2)) ** 0.5
        print(f'Rp[n-1] = {Rp_list[n-1]}')
        print(f'dRp = {dRp}')

        # Заполняет массивы глубины и концентрации для дальнейших построений, рассчитывая при этом С
        for i in range (1, n):
            x_list[i] = i * dx
            C_list[i] = (self.Q / (dRp * (2 * 3.14) ** 0.5)) * np.exp(-(x_list[i] - Rp_list[n - 1]) ** 2 / (2 * dRp ** 2))

        # Возвращает индекс наибольшего элемента массива C_list и находит наибольший аргумент по этому индексу
        C0 = C_list[np.argmax(C_list)]
        print(f'C0 = {C0}')

        fig, axes = plt.subplots()

        # Построение графика зависимости
        axes.plot(x_list, C_list, color='blue', label='C(x)')
        plt.title('Распеделение концентрации по глубине', pad = 20)
        plt.ylabel('С, см⁻³')
        plt.xlabel('х, см')

        axes.set_xlim(0, self.x)
        axes.set_ylim(0)

        plt.legend(loc=5)

        # Добавление дополнительной ссетки
        axes.grid(which='major', color = '#666666')
        axes.minorticks_on()
        axes.grid(which='minor', color = 'gray', linestyle = ':')

        # Вывод отрисовки на экран
        plt.show()

def main():
    # Объявляем экземпляр класса для отрисовки, выводим необходимые расчетные параметры на экран, и запускаем непосредственно функцию отрисовки графика
    new2 = Draw_graph()
    print(f'Se = {new2.Se(new2.E)}')
    print(f'Sn = {new2.Sn(new2.E)}')
    new2.draw()

# запускаем тело программы
if __name__ == "__main__":
    main()