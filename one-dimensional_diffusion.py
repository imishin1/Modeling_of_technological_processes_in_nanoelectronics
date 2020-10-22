import math
import numpy as np
import matplotlib.pyplot as plt

# Mishin Ilya
# The prosecc of deffusion
# Substrate - Si <100> p-type
# N(As) = 4E+16 cm^-3
# admixture - As
# T1 = 1050 C
# t1 = 15 min
# T2 = 950 C
# t2 = 15 min
# x = 0.45 mkm

class Physical_quantities:
    def __init__(self):
        self.q = 1.6e-19 # charge in Si
        self.k = 8.617e-5 # Boltzmann constant, eV

class Initial_param:
    def __init__(self):
        self.N_As = 4E+16 # sm^-3
        self.T_zagonka = 1050 + 273 # K
        self.t_zagonka = 15 * 60 # s
        self.T_razgonka = 950 + 273 # K
        self.t_razgonka = 15 * 60 # s
        self.x = 0.8 # mkm - вводим вручную
        self.Ea = 2.42 # eV
        self.D0 = 8.5E-2 # sm^2 / 
        self.C0 = 2E21

class Culculation_param:
    def __init__(self):
        Physical_quantities.__init__(self)
        Initial_param.__init__(self)
    
    #find Eg in eV
    def find_Eg(self, temperature):
        return 1.17 - 4.73E-4 * (temperature ** 2) / (temperature + 636)

    #find Nc in sm^-3
    def find_Nc(self, temperature):
        return 6.2E+15 * (temperature ** (3/2))   

    #find Nv in sm^-3
    def find_Nv(self, temperature):
        return 3.5E+15 * (temperature ** (3/2)) 

    def find_ni(self, temperature):
        return ((self.find_Nc(temperature) * self.find_Nv(temperature) / 2) ** 0.5) * np.exp(-self.find_Eg(temperature) / (2 * self.k * temperature))

    def find_Diff(self, n, temperature):
        return 0.66 * np.exp(-3.44 / self.k / temperature) + 12 * (n / self.find_ni(temperature)) * np.exp(-4.05 / self.k / temperature)

class Draw_graph(Culculation_param):
    def draw(self):
        n = 200
        C_list = [0] * n
        x_list = [0] * n
        a_list = [0] * n
        b_list = [0] * n 
        d_list = [0] * n
        delta_list = [0] * n
        lambda_list = [0] * n
        r_list = [0] * n
        Cn_list = [0] * n
        dt = 1
        dx = self.x / (10000 * n)

        # zagonka
        # for left bord index - 0:
        b_list[0] = 0
        d_list[0] = 0
        r_list[0] = self.C0
        a_list[0] = 1
        delta_list[0] = -d_list[0] / a_list[0]
        lambda_list[0] = r_list[0] / a_list[0]

        # for right bord index - n:
        b_list[n-1] = 0
        d_list[n-1] = 0
        r_list[n-1] = 0
        a_list[n-1] = 1

        for i in range(0, n):
            x_list[i] = dx * i
        for j in range (0, self.t_zagonka):
            for i in range (1, n - 1):
                # тут до n-2 не будет работать
                b_list[i] = 1
                d_list[i] = 1
                a_list[i] = (- (2 + (dx ** 2) / (self.find_Diff(C_list[i], self.T_zagonka) * dt)))
                r_list[i] = (- (dx ** 2) / (self.find_Diff(C_list[i], self.T_zagonka) * dt) * C_list[i])
            for i in range (1, n):
                delta_list[i] = (- d_list[i] / (a_list[i] + b_list[i] * delta_list[i-1]))
                lambda_list[i] = ((r_list[i] - b_list[i] * lambda_list[i-1]) / (a_list[i] + b_list[i] * delta_list[i-1]))
            C_list[n-1] = lambda_list[n-1]
        
            for i in range(n-2, -1, -1):
                C_list[i] = C_list[i+1] * delta_list[i] + lambda_list[i]

        fig, axes = plt.subplots()
        axes.plot(x_list, C_list)

        # razgonka
        # for left bord index - 0:
        b_list[0] = 0
        d_list[0] = 1
        r_list[0] = 0
        a_list[0] = -1
        delta_list[0] = -d_list[0] / a_list[0]
        lambda_list[0] = r_list[0] / a_list[0]

        # for right bord index - n:
        b_list[n-1] = 0
        d_list[n-1] = 0
        r_list[n-1] = 0
        a_list[n-1] = 1

        for j in range (0, self.t_razgonka):
            for i in range (1, n - 1):
                # тут до n-2 не будет работать
                b_list[i] = 1
                d_list[i] = 1
                a_list[i] = (- (2 + (dx ** 2) / (self.find_Diff(C_list[i], self.T_razgonka) * dt)))
                r_list[i] = (- (dx ** 2) / (self.find_Diff(C_list[i], self.T_razgonka) * dt) * C_list[i])
            for i in range (1, n):
                delta_list[i] = (- d_list[i] / (a_list[i] + b_list[i] * delta_list[i-1]))
                lambda_list[i] = ((r_list[i] - b_list[i] * lambda_list[i-1]) / (a_list[i] + b_list[i] * delta_list[i-1]))
            C_list[n-1] = lambda_list[n-1]
        
            for i in range(n-2, -1, -1):
                C_list[i] = C_list[i+1] * delta_list[i] + lambda_list[i]

        axes.plot(x_list, C_list)

        axes.set_xlim(0)
        plt.show()

def main():
    As = Draw_graph()
    As.draw()

if __name__ == "__main__":
    main()