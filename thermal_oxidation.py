# импортируем расчетные библиотеки
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from one_dimensional_diffusion import Physical_quantities, Culculation_param

# Si <100> p-type
# xi = 0,1 mkm
# t = 30 min
# C1 = 1E15 cm^-3
# C2 = 4E17 cm^-3
# O2
# T = 1050 + 273 K
# P = 1.5 atm

class Oxidation_param:
    def __init__(self):
        self.P = 1.5 # atm
        self.T = 1050 + 273 # K
        self.t = 30 # min
        self.x = 0.1 * 2.4 # mkm
        self.C1 = 1E15 # cm^-3
        self.C2 = 4E17 # cm^-3
        self.A0_parabolic = 12.9 # mkm^2 / min
        self.Ea_parabolic = 1.23 # mkm^2 / min
        self.A0_linear = 1.4E5 # mkm / min
        self.Ea_linear = 2 # mkm / min

class Culculation_param:
    def __init__(self):
        Physical_quantities.__init__(self)
        Oxidation_param.__init__(self)
    
    # find Eg in eV
    def find_Eg(self, temperature):
        return 1.17 - 4.73E-4 * (temperature ** 2) / (temperature + 636)

    # find Nc in sm^-3
    def find_Nc(self, temperature):
        return 6.2E+15 * (temperature ** (3/2))   

    # find Nv in sm^-3
    def find_Nv(self, temperature):
        return 3.5E+15 * (temperature ** (3/2)) 

    # find ni in sm^-3
    def find_ni(self, temperature):
        return ((self.find_Nc(temperature) * self.find_Nv(temperature) / 2) ** 0.5) * np.exp(-self.find_Eg(temperature) / (2 * self.k * temperature))
    
    # Функция расчета коэффициентов от температуры
    def find_B_or_B_A(self, temperature, A0, Ea):
        return A0 * np.exp(-Ea / (self.k * temperature))

    # Функция расчета линейного коэффициента от давления
    def find_B_A_from_P(self, B_A):
        return B_A * self.P ** (0.75)
    
    # Функция расчета коэффициентов от ориентации пластинки при <100>
    def find_B_A_from_orientation(self, B_A):
        return B_A / 1.68

    # Функция расчта линейного коэффициента от концентрации
    def find_B_A_from_C(self, B_A, C, temperature):
        def gamma_l(temperature):
            return 2620 * np.exp(-1.1 / self.k / temperature)
        
        def Vm(temperature, C):
            def find_Ei(temperature):
                return self.find_Eg(temperature) / 2 - self.k * temperature / 4

            def find_C_minus (temperature):
                return np.exp((find_Ei(temperature) + 0.57 - self.find_Eg(temperature)) /self.k / temperature)
            
            def find_C_2minus(temperature):
                return np.exp((2 * find_Ei(temperature) + 1.25 - 3 * self.find_Eg(temperature)) / self.k / temperature)

            def find_C_plus(temperature):
                return np.exp((0.35 - find_Ei(temperature)) / self.k / temperature)
            
            return (1 + find_C_plus(temperature) * (self.find_ni(temperature) / C) + find_C_minus(temperature) * (C / self.find_ni(temperature)) \
            + find_C_2minus(temperature) * (C / self.find_ni(temperature)) ** 2) \
            / (1 + find_C_2minus(temperature) + find_C_minus(temperature) + find_C_plus(temperature))
        
        return B_A * (1 + gamma_l(temperature) * (Vm(temperature, C) - 1))

    # Функция расчета параболического коэффициента от концентрации
    def find_B_from_C(self, B, C, temperature):
        def gamma_p(temperature):
            return 9.63E-16 * np.exp(2.83 / self.k / temperature)
        
        return B * (1 + gamma_p(temperature) * C ** 0.22)

    # Функция расчета толщины окисла
    def x_t(self, t, B_A, B, xi):
        A = B / B_A
        def talu(A, B, xi):
            return (xi ** 2 + A * xi) / B
        
        return A / 2 * ((1 + (t + talu(A, B, xi)) / (A ** 2 / 4 / B))**0.5 - 1)

    # Функция нахождения времени окисления по заданной толщине окисления
    def find_time(self, B_A, B, x, talu):
        A = B / B_A
        return (A ** 2)  / 4 / B * ((1 + 2 * x / A) ** 2 - 1) - talu


class Draw_graph(Culculation_param):
    def draw(self):
        # Объявим массивы для времени и концентрации
        t_list = list()
        t1_list = list()
        x_list = list()
        x1_list = list()

        # Проведем расчет первого слоя
        # Расчитаем значения коэффициентов в зависимости от заданных условий
        B_A = self.find_B_A_from_C(self.find_B_A_from_orientation((self.find_B_A_from_P(self.find_B_or_B_A(self.T, self.A0_linear, self.Ea_linear)))), self.C1, self.T)
        B = self.find_B_from_C(self.find_B_or_B_A(self.T, self.A0_parabolic, self.Ea_parabolic), self.C1, self.T)

        # Расчитаем длительность окисления первого слоя
        t1 = self.find_time(B_A, B, self.x, 0)

        # Выберем шаг для первого слоя
        n = 10000  
        dt1 = t1 / n

        # Расчитаем x(t) для первого слоя
        xi = 0
        for i in np.arange (0, t1, dt1):
            t1_list.append(i)
            x_t = (self.x_t(dt1, B_A, B, xi))
            x1_list.append(x_t)
            xi = x_t

        # Проведем расчет второго слоя
        # Расчитаем значения коэффициентов в зависимости от заданных условий
        B_A = self.find_B_A_from_C(self.find_B_A_from_orientation((self.find_B_A_from_P(self.find_B_or_B_A(self.T, self.A0_linear, self.Ea_linear)))), self.C2, self.T)
        B = self.find_B_from_C(self.find_B_or_B_A(self.T, self.A0_parabolic, self.Ea_parabolic), self.C2, self.T)

        # Выберем шаг для второго слоя
        m = 10000
        dt = self.t / m

        # Расчитаем x(t) для второго слоя
        for i in np.arange (0, self.t, dt):
            t_list.append(t1 + i)
            x_t = (self.x_t(dt, B_A, B, xi))
            x_list.append(x_t)
            xi = x_t

        # Объявим форму для рисования и оси
        fig, axes = plt.subplots()
        
        # Добавим на график зависимости для окисления обоих слоев
        axes.plot(t1_list, x1_list, color='blue', label='1 слой')
        axes.plot(t_list, x_list, color='orange', label='2 слой')

        # Ограничим оси в нужном диапозоне
        axes.set_xlim(0, t1 + self.t)
        axes.set_ylim(0, x_list[-1])

        # Добавим название для графика и осей
        plt.title('Зависимость толщины окисла от времени',)
        plt.xlabel('t, мин')
        plt.ylabel('х, мкм')
        plt.legend(loc=5)

        # Добавление дополнительной ссетки
        axes.grid(which='major', color = '#666666')
        axes.minorticks_on()
        axes.grid(which='minor', color = 'gray', linestyle = ':')

        # Выведем график
        plt.show()

def main():
    # Объявляем экземпляр класса для отрисовки
    therm_oxid = Draw_graph()
    therm_oxid.draw()

# запускаем тело программы
if __name__ == "__main__":
    main()