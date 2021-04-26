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

class myarray(np.ndarray):
    def __new__(cls, *args, **kwargs):
        return np.array(*args, **kwargs).view(myarray)
    def index(self, value):
        return np.where(self == value)

class Initial_param:
    def __init__(self):
        self.x = 0.75 # mkm
        self.N = 7E15 # sm^-3
        self.talu = 1E-4 # s
        #self.lambd = 0.6 # mkm
        self.N_plus = self.N * 100 # sm^-3
        self.S = 5E4 # sm/s
        self.mu_n = 3900 # sm^2/(V*s)
        self.mu_p = 1900 # sm^2/(V*s)
        self.epsilond = 16
        self.A = 1 # sm^2
        self.Eg = 0.67 # eV
        self.lam = 1.24 / self.Eg
        #self.n = 4 + 0.001106337 - 0.00314503 / self.lambd + 0.492812 / (self.lambd ** 2) - 0.601906 / (self.lambd ** 3) + 0.982897 / (self.lambd ** 4)
        #self.Re = (self.n - 1) ** 2 / (self.n + 1) ** 2
        #self.k = 1.4
        #self.alfa = 4 * np.pi * self.k / (self.lambd * 1E-4)
        self.d = 2 # mkm
        self.I0 = 1E15
        self.U = 0
        self.ni = 24000000000000
        self.dp = 50
        self.dn = 100

class Culculation_param:
    def __init__(self):
        Physical_quantities.__init__(self)
        Initial_param.__init__(self)
    
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
    def draw(self):
        m = 1000
        x_list = List_of_param(m).new_list
        G_list = List_of_param(m).new_list
        dx = self.d * 1E-4 / (m - 1)

        fk = 0.026 * np.log(self.N * self.N_plus / (self.ni ** 2))
        Nb = self.N * self.N_plus/ (self.N + self.N_plus)
        Wp = ((2 * self.epsilond * self.epsilond_0 * Nb * (fk - self.U) / self.q) ** 0.5) / self.N_plus
        Wn = ((2 * self.epsilond * self.epsilond_0 * Nb * (fk - self.U) / self.q) ** 0.5) / self.N
        W = Wn + Wp

        sp = int((self.x * 1E-4 - Wp) * m / (self.d * 1E-4))
        sn = int((self.d * 1E-4 - self.x * 1E-4 - Wn) * m / (self.d * 1E-4))
        sw = m - sn - sp

        Gp_list = List_of_param(sp).new_list
        Gn_list = List_of_param(sn).new_list
        xn_list = List_of_param(sn).new_list
        xp_list = List_of_param(sp).new_list
        dpn_list = List_of_param(sn).new_list
        dnp_list = List_of_param(sp).new_list
        dpn1_list = List_of_param(sn).new_list
        dnp1_list = List_of_param(sp).new_list
        
        lambd_list = list()
        S_list = list()

        Gp_list = G_list[0 : sp]
        Gn_list = G_list[m - sn : m]
        xp_list = x_list[0 : sp]
        xn_list = x_list[m - sn : m]

        dpn_list = Gn_list * self.talu
        dnp_list = Gp_list * self.talu

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

        for lambd in np.arange(0.2, 0.8, 0.1):
            for i in range (0, m):
                x_list[i] = i * dx
                G_list[i] = self.find_G_x(x_list[i], self.find_k(lambd), lambd)

            for j in range (0, 200):
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
                
            k = self.find_k(lambd)
            alfa = self.find_alfa(k, lambd)
            jn = self.q * self.dn * (dnp_list[sp - 2] - dnp_list[sp - 1]) / dx
            #jp = self.q * self.dp * (dpn_list[1] - dpn_list[0]) / dx
            #jw = self.q * self.I0 * (1 - self.find_Re(lambd)) * np.exp(-alfa * (self.x - Wp)) - self.q * self.I0 * (1 - self.find_Re(lambd)) * np.exp(-alfa * (self.x + Wn))
            #jph = jp + jn + jw
            P = self.I0 * self.q * 1.24 / lambd
            S = abs(jn / P)
            #print(lambd)
            #print(alfa)
            #print(f'S = {S}')
            lambd_list.append(lambd)
            S_list.append(S)

        '''# Прогоночка для n
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

        for lambd in np.arange(0.2, 0.8, 0.1):
            for j in range (0, 20):
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

                dpn1_list = dpn_list'''

        fig, axes = plt.subplots()

        #axes.plot(lambd_list, S_list)
        #axes.plot(x_list, G_list)
        axes.plot(xp_list, dnp_list)
        #axes.plot(xn_list, dpn_list)
        #axes.plot(xn_list, Gn_list)
        #axes.plot(xp_list, Gp_list)
        #axes.plot([self.x * 1E-4 + Wn, self.x * 1E-4 + Wn], [G_list[0], G_list[-1]])
        #axes.plot([self.x * 1E-4 - Wp, self.x * 1E-4 - Wp], [G_list[0], G_list[-1]])

        #axes.set_xlim(0, self.d * 1E-4)
        #axes.set_ylim(0)

        #plt.title('Распеделение концентрации по глубине залегания', pad = 20)
        #plt.xlabel('х, см')
        #plt.ylabel('С, см⁻³')

        # Добавление дополнительной ссетки
        axes.grid(which='major', color = '#666666')
        axes.minorticks_on()
        axes.grid(which='minor', color = 'gray', linestyle = ':')

        plt.show()

def main():
    # объявляем объем и запускаем функции рисования
    Ge = Draw_graph()
    Ge.draw()

# запускаем тело программы
if __name__ == "__main__":
    main()