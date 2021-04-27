# импортируем расчетные библиотеки
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class Physical_quantities:
    def __init__(self):
        self.q = 1.6e-19 # charge in Si
        self.k = 8.617e-5 # Boltzmann constant, eV
        self.epsilond_0 = 8.85E-14 # F/sm

class List_of_param():
    # Create a necessary lists for calculating param and drawing the graphs
    def __init__(self, n):
        # создание нулевого массива длинной n
        self.new_list = np.zeros(n)

class Initial_param:
    def __init__(self):
        self.x = 0.75 # mkm
        self.N = 7E15 # sm^-3
        self.talu = 1E-4 # s
        self.N_plus = self.N * 100 # sm^-3
        self.S = 5E4 # sm/s
        self.epsilond = 16
        self.Eg = 0.67 # eV
        self.d = 5 # mkm
        self.I0 = 1E15 # Vt/sm
        self.ni = 24000000000000 # sm^-3
        self.dp = 50 # sm^2/s
        self.dn = 100 # sm^2/s

class Culculation_param:
    def find_n(self, lambd):
        return 4 + 0.001106337 - 0.00314503 / lambd + 0.492812 / (lambd ** 2) - 0.601906 / (lambd ** 3) + 0.982897 / (lambd ** 4)

    def find_k(self, lambd):
        return -8342.9 * lambd ** 6 + 27697 * lambd ** 5 - 37013 * lambd ** 4 + 25399 * lambd ** 3 - 9410.2 * lambd ** 2 + 1773.8 * lambd - 128.92
    
    def find_Re(self, lambd):
        return (self.find_n(lambd) - 1) ** 2 / (self.find_n(lambd) + 1) ** 2

    def find_alfa(self, k, lambd):
        return 4 * np.pi * self.find_k(lambd) / (lambd * 1E-4)

    def find_G_x(self, x, k, lambd):
        return self.I0 * (1 - self.find_Re(lambd)) * self.find_alfa(k, lambd) * np.exp(-x * self.find_alfa(k, lambd))

class Draw_graph(Culculation_param):
    def __init__(self, U):
        Physical_quantities.__init__(self)
        Initial_param.__init__(self)
        self.Sn_list = list()
        self.Sp_list = list()
        self.Sw_list = list()
        self.lambd_list = list()
        self.U = U
        # Вводим область от 0.2 до 0.76 ввиду того, что апроксимация для функции параметра k была выведена для этих границ (сайт дает только такие)
        self.lambd_min = 0.2
        self.lambd_max = 0.76
        self.lambd_step = 0.01

    def count_S(self):
        m = 100

        # Задаем нулевые массивы для глубины и генерации, а также рассчитываем величину приращения dx
        x_list = List_of_param(m).new_list
        G_list = List_of_param(m).new_list
        dx = self.d * 1E-4 / (m - 1)

        # Рассчитываем контактную разность потенциалов и ширину ОПЗ (в том числе с каждой стороны от x)
        fk = 0.026 * np.log(self.N * self.N_plus / (self.ni ** 2))
        Nb = self.N * self.N_plus/ (self.N + self.N_plus)
        Wp = ((2 * self.epsilond * self.epsilond_0 * Nb * (fk - self.U) / self.q) ** 0.5) / self.N_plus
        Wn = ((2 * self.epsilond * self.epsilond_0 * Nb * (fk - self.U) / self.q) ** 0.5) / self.N
        W = Wn + Wp
    
        # Рассчитываем число индексов для каждой области (нужно будет, чтобы разбить на разные массивы каждой области)
        sp = int((self.x * 1E-4 - Wp) * m / (self.d * 1E-4))
        sn = int((self.d * 1E-4 - self.x * 1E-4 - Wn) * m / (self.d * 1E-4))
        sw = m - sn - sp

        # Задаем необходимые нулевые массивы
        Gp_list = List_of_param(sp).new_list
        Gn_list = List_of_param(sn).new_list
        xn_list = List_of_param(sn).new_list
        xp_list = List_of_param(sp).new_list
        dpn_list = List_of_param(sn).new_list
        dnp_list = List_of_param(sp).new_list
        dpn1_list = List_of_param(sn).new_list
        dnp1_list = List_of_param(sp).new_list

        # Разбиваем массив с общей генерацией и глубиной на частные массивы для p и n области
        Gp_list = G_list[0 : sp]
        Gn_list = G_list[m - sn : m]
        xp_list = x_list[0 : sp]
        xn_list = x_list[m - sn : m]

        dpn_list = Gn_list * self.talu
        dnp_list = Gp_list * self.talu

        # --------------------------------------------------------------------------------------------------

        # Прогоночка p+
        a_list = List_of_param(sp).new_list
        d1_list = List_of_param(sp).new_list
        b_list = List_of_param(sp).new_list
        r_list = List_of_param(sp).new_list
        de_list = List_of_param(sp).new_list
        la_list = List_of_param(sp).new_list

        # Левые границы
        b_list[0] = 1
        a_list[0] = self.S * dx / self.dn - 1
        d1_list[0] = 0
        r_list[0] = 0

        de_list[0] = -d1_list[0] / a_list[0]
        la_list[0] = r_list[0] / a_list[0]

        # Правые границы
        b_list[sp - 1] = 0
        a_list[sp - 1] = 1
        d1_list[sp - 1] = 0
        r_list[sp - 1] = 0

        for lambd in np.arange(self.lambd_min, self.lambd_max, self.lambd_step):
            for i in range (0, m):
                x_list[i] = i * dx
                G_list[i] = self.find_G_x(x_list[i], self.find_k(lambd), lambd)

            for j in range (0, 100):
                for i in range (1, sp - 1):
                    d1_list[i] = 1
                    a_list[i] = -2 - (dx * dx / (self.dn * self.talu))
                    b_list[i] = 1
                    r_list[i] = -dx * dx * Gp_list[i] / self.dn
                
                for i in range(1, sp):
                    de_list[i] = -(d1_list[i] / (a_list[i] + b_list[i] * de_list[i - 1]))
                    la_list[i] = (r_list[i] - b_list[i] * la_list[i - 1]) / (a_list[i] + b_list[i] * de_list[i - 1])

                dnp_list[sp - 1] = la_list[sp - 1]

                for i in range (sp - 2, -1, -1):
                    dnp_list[i] = de_list[i] * dnp_list[i + 1] + la_list[i]

                dnp1_list = dnp_list
            
            # Рассчитываем ток и относительную фоточувствительность. Формируем массивы спектра и фоточувствительности каждоый области
            k = self.find_k(lambd)
            alfa = self.find_alfa(k, lambd)
            jn = self.q * self.dn * (dnp_list[sp - 2] - dnp_list[sp - 1]) / dx
            P = self.I0 * self.q * 1.24 / lambd
            S = abs(jn / P)
            self.lambd_list.append(lambd)
            self.Sn_list.append(S)

        # --------------------------------------------------------------------------------------------------

        # Прогоночка для n
        a_list = List_of_param(sn).new_list
        d1_list = List_of_param(sn).new_list
        b_list = List_of_param(sn).new_list
        r_list = List_of_param(sn).new_list
        de_list = List_of_param(sn).new_list
        la_list = List_of_param(sn).new_list

        # Правые границы
        b_list[sn - 1] = 1
        a_list[sn - 1] = self.S * dx / self.dp - 1
        d1_list[sn - 1] = 0
        r_list[sn - 1] = 0

        # Левые границы
        b_list[0] = 0
        a_list[0] = 1
        d1_list[0] = 0
        r_list[0] = 0

        de_list[0] = -d1_list[0] / a_list[0]
        la_list[0] = r_list[0] / a_list[0]

        for lambd in np.arange(self.lambd_min, self.lambd_max, self.lambd_step):
            for i in range (0, m):
                x_list[i] = i * dx
                G_list[i] = self.find_G_x(x_list[i], self.find_k(lambd), lambd)

            for j in range (0, 100):
                for i in range (1, sn - 1):
                    d1_list[i] = 1
                    a_list[i] = -2 - (dx * dx / (self.dp * self.talu))
                    b_list[i] = 1
                    r_list[i] = -dx * dx * Gn_list[i] / self.dp
                
                for i in range(1, sn):
                    de_list[i] = -(d1_list[i] / (a_list[i] + b_list[i] * de_list[i - 1]))
                    la_list[i] = (r_list[i] - b_list[i] * la_list[i - 1]) / (a_list[i] + b_list[i] * de_list[i - 1])

                dpn_list[sn - 1] = la_list[sn - 1]

                for i in range (sp - 2, -1, -1):
                    dpn_list[i] = de_list[i] * dpn_list[i + 1] + la_list[i]

                dpn1_list = dpn_list

            # Рассчитываем ток и относительную фоточувствительность. Формируем массивы спектра и фоточувствительности каждоый области
            k = self.find_k(lambd)
            alfa = self.find_alfa(k, lambd)
            P = self.I0 * self.q * 1.24 / lambd
            jp = self.q * self.dp * (dpn_list[1] - dpn_list[0]) / dx
            jw = self.q * self.I0 * (1 - self.find_Re(lambd)) * np.exp(-alfa * (self.x * 1E-4 + Wp)) - \
            self.q * self.I0 * (1 - self.find_Re(lambd)) * np.exp(-alfa * (self.x * 1E-4 - Wn))
            Sp = abs(jp / P)
            Sw = abs(jw / P)
            self.Sp_list.append(Sp)
            self.Sw_list.append(Sw)
        
        # тут просто сложение элементов трех массивов, чтоб определить общую S
        self.S_list = list(map(lambda x, y, z: x + y + z, self.Sp_list, self.Sw_list, self.Sn_list))

def draw_S(lambd_list, lambd_min, lambd_max, Sn_list, Sp_list, Sw_list, S_list):
    fig, axes = plt.subplots()

    axes.plot(lambd_list, Sn_list, label='p+')
    axes.plot(lambd_list, Sp_list, label='n')
    axes.plot(lambd_list, Sw_list, label='w')
    axes.plot(lambd_list, S_list, label='summ')
    axes.set_xlim(lambd_min, lambd_max)
    axes.set_ylim(0)
    plt.legend(loc=6)

    plt.title('Спектральная характеристика относительной \n фоточувствительности фотодиода', pad=10)
    plt.xlabel('λ, мм')
    plt.ylabel('S')

    # Добавление дополнительной ссетки
    axes.grid(which='major', color = '#666666')
    axes.minorticks_on()
    axes.grid(which='minor', color = 'gray', linestyle = ':')

    plt.show()

def draw_S_V(lambd_list, lambd_min, lambd_max, S_V1_list, S_V2_list, S_V3_list, S_V4_list, V1, V2, V3, V4):
    fig, axes = plt.subplots()

    axes.plot(lambd_list, S_V1_list, label=f'Uвн = {V1} В')
    axes.plot(lambd_list, S_V2_list, label=f'Uвн = {V2} В')
    axes.plot(lambd_list, S_V3_list, label=f'Uвн = {V3} В')
    axes.plot(lambd_list, S_V4_list, label=f'Uвн = {V4} В')
    axes.set_xlim(lambd_min, lambd_max)
    axes.set_ylim(0)
    plt.legend(loc=6)

    plt.title('Спектральная характеристика относительной фоточувствительности \n фотодиода от внешнего смещения', pad=10)
    plt.xlabel('λ, мм')
    plt.ylabel('S')

    # Добавление дополнительной ссетки
    axes.grid(which='major', color = '#666666')
    axes.minorticks_on()
    axes.grid(which='minor', color = 'gray', linestyle = ':')

    plt.show()

def main():
    Ge = Draw_graph(0)
    Ge.count_S()
    draw_S(Ge.lambd_list, Ge.lambd_min, Ge.lambd_max, Ge.Sn_list, Ge.Sp_list, Ge.Sw_list, Ge.S_list)
    Ge_V1 = Draw_graph(0)
    Ge_V2 = Draw_graph(-0.5)
    Ge_V3 = Draw_graph(-1)
    Ge_V4 = Draw_graph(-1.5)
    Ge_V1.count_S()
    Ge_V2.count_S()
    Ge_V3.count_S()
    Ge_V4.count_S()
    draw_S_V(Ge.lambd_list, Ge.lambd_min, Ge.lambd_max, Ge_V1.S_list, Ge_V2.S_list, Ge_V3.S_list, Ge_V4.S_list, Ge_V1.U, Ge_V2.U, Ge_V3.U, Ge_V4.U)

# запускаем тело программы
if __name__ == "__main__":
    main()