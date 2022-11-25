import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

"""
State Space Formulation

"""


def dR(y1, y2, y1_new, y2_new, h):
    return  y1 + h*(0.01*y1_new**2 - 0.02*y2_new**3) - y1_new
    


def dJ(y1, y2, y1_new, y2_new, h):

    return  y2 + h*(0.01*(y1_new**2)*(y2_new**2)) - y2_new


def jacobian(y1, y2, y1_new, y2_new, h):

    J = np.ones([2, 2])
    d = 1e-9

    J[0, 0] = (dR(y1, y2, (y1_new+d), y2_new, h) -
               dR(y1, y2, y1_new, y2_new, h))/d
    J[0, 1] = (dR(y1, y2, y1_new, (y2_new+d), h) -
               dR(y1, y2, y1_new, y2_new, h))/d

    J[1, 0] = (dJ(y1, y2, (y1_new+d), y2_new, h) -
               dJ(y1, y2, y1_new, y2_new, h))/d
    J[1, 1] = (dJ(y1, y2, y1_new, (y2_new+d), h) -
               dJ(y1, y2, y1_new, y2_new, h))/d

    return J


def NewtRhap(y1, y2, y1_guess, y2_guess, h):

    S_old = np.ones([2, 1])
    S_old[0] = y1_guess
    S_old[1] = y2_guess

    F = np.ones([2, 1])

    error = 9e9
    tol = 1e-9
    aplha = 1


    while error > tol:

        J = jacobian(y1, y2, S_old[0], S_old[1], h)

        F[0] = dR(y1, y2, S_old[0], S_old[1], h)
        F[1] = dJ(y1, y2, S_old[0], S_old[1], h)

        S_new = S_old - aplha*(np.matmul(inv(J), F))

        error = np.max(np.abs(S_new - S_old))

        S_old = S_new

    return [S_new[0], S_new[1]]


def implicit_Euler(inty1, inty2, tspan, dt):

    t = np.arange(0, tspan, dt)
    y1 = np.zeros(len(t))
    y2 = np.zeros(len(t))

    y1[0] = inty1
    y2[0] = inty2

    y1_guess = 5
    y2_guess = 5

    for i in range(1, len(t)):

        y1[i], y2[i] = NewtRhap(
            y1[i-1], y2[i-1], y1_guess, y2_guess, dt)

        y1_guess = y1[i]
        y2_guess = y2[i]

    return [t, y1, y2]


t, y1, y2 = implicit_Euler(0, 1, 50, 0.5)

plt.plot(t, y1, 'b')
plt.plot(t, y2, 'g')
plt.legend(["graph of y1", "graph of y2"])
plt.show()
