import math
from copy import deepcopy

import pandas as pd
import matplotlib.pyplot as plt


# Funkcje obsługujące działania na macierzach/wektorach potrzebne do metody LU
def forward_substitution(A, b):
    x = []  # dla układu równań Ax = b
    for i in range(len(b)):
        x.append(b[i])
        for j in range(i):
            val = x[i] - (A[i][j] * x[j])
            x[i] = val
        x[i] /= A[i][i]
    return x


def backward_substitution(A, b):
    x = []  # dla układu równań Ax = b
    m = len(b) - 1
    for i in range(len(b)):
        x.append(b[m - i])
        for j in range(i):
            val = x[i] - (A[m - i][m - j] * x[j])
            x[i] = val
        x[i] /= A[m-i][m-i]
    x.reverse()
    return x


def sub_vectors(x1, x2):
    output = deepcopy(x1)
    for i in range(len(x1)):
        output[i] -= x2[i]
    return output


def multiply_matrix_by_vector(A, b):
    multiplied_vector = []  # Mnożenie macierzy przez pionowy wektor
    for i in range(len(b)):
        value = 0
        for j in range(len(A[i])):
            value += A[i][j] * b[j]
        multiplied_vector.append(value)
    return multiplied_vector  # wynikiem jest nowy pionowy wektor


def norm(v):  # obliczanie normy zgodnie ze wzorem ( p = 2 )
    norm_value = 0
    for i in range(len(v)):
        norm_value += (v[i] * v[i])
    return math.sqrt(norm_value)


# Nowe Funkcje
def factorization_LU_with_pivoting(A, x, b):
    #start = time.time()
    U = deepcopy(A)  # Utworzenie macierzy L i U
    L = [0] * len(A)  # macierz I
    for i in range(len(A)):
        L[i] = [0] * len(A)
    for i in range(len(A)):
        for j in range(len(A)):
            if i == j:
                L[i][j] = 1

    P = [0] * len(A)  # macierz I - Permutation Matrix
    for i in range(len(A)):
        P[i] = [0] * len(A)
    for i in range(len(A)):
        for j in range(len(A)):
            if i == j:
                P[i][j] = 1

    for i in range(len(A)-1):
        # pivoting
        # znajdywanie pivota
        pivot = abs(U[i][i])
        ind = i
        for j in range(i, len(A)):
            val = abs(U[j][i])
            if val > pivot:
                pivot = val
                ind = j
        #ind = ind + i - 1
        # zamienianie rzędów
        # U
        for j in range(i, len(A)):
            temp = U[i][j]
            U[i][j] = U[ind][j]
            U[ind][j] = temp
        # L
        for j in range(0, i):
            temp = L[i][j]
            L[i][j] = L[ind][j]
            L[ind][j] = temp
        # P
        for j in range(0, len(A)):
            temp = P[i][j]
            P[i][j] = P[ind][j]
            P[ind][j] = temp

        # Standardowe LU
        for j in range(i+1,len(A)):
            L[j][i] = U[j][i]/U[i][i]
            for k in range(i,len(A)):
                U[j][k] = U[j][k] - L[j][i] * U[i][k]

    #  rozwiązanie układu równań Ly = b
    b = multiply_matrix_by_vector(P, b)
    y = forward_substitution(L, b)
    #  rozwiązanie układu równań Ux = y
    x = backward_substitution(U, y)  # wynik
    #  sprawdzenie normy z wektora residuum
    help1 = multiply_matrix_by_vector(A, x)
    res = sub_vectors(help1, b)
    norm_of_residuum = norm(res)
    #alg_time = end-start
    #print("Faktoryzacja LU")
    #print("Norma z residuum: " + str(norm_of_residuum))
    #print("Czas: " + str(end - start))
    #print(x)
    return x


def lagrange_interpolate_point(nodes_x, nodes_y, x):
    size = len(nodes_x)
    interpolated_value = 0
    for i in range(0, size):
        fi = 1
        for j in range(0, size):
            if i != j:
                fi *= (x - nodes_x[j]) / (1.0 * (nodes_x[i] - nodes_x[j]))
        interpolated_value += (fi * nodes_y[i])
    return interpolated_value


def lagrange(input_x, nodes_x, nodes_y):
    interpolated_values = []
    for x in input_x:
        interpolated_values.append(lagrange_interpolate_point(nodes_x, nodes_y, x))
    return interpolated_values


def spline_interpolate_point(nodes_x, factors, x):
    for i in range(len(nodes_x)-1):
        if nodes_x[i] <= x <= nodes_x[i + 1]:
            h = x - nodes_x[i]
            interpolated_value = factors[4 * i] + (factors[4 * i + 1] * h)
            interpolated_value += (factors[4 * i + 2] * h * h) + (factors[4 * i + 3] * h * h * h)
            return interpolated_value
    return 0


def spline(input_x, nodes_x, nodes_y):
    n = len(nodes_x)  # ilość węzłów
    matrix_size = 4 * (n-1)
    A = [0] * matrix_size  # macierz i wektory reprezentujące układ równań
    for i in range(matrix_size):
        A[i] = [0] * matrix_size
    x = [1] * matrix_size
    b = [0] * matrix_size

    for i in range(0, n-1):
        h = nodes_x[i + 1] - nodes_x[i]
        # Si(xi) = f(xi)
        A[4*i][4*i] = 1
        b[4*i] = nodes_y[i]
        # Si(xi+1) = f(xi+1)
        A[4*i+1][4*i] = 1
        A[4 * i + 1][4 * i + 1] = h
        A[4 * i + 1][4 * i + 2] = h * h
        A[4 * i + 1][4 * i + 3] = h * h * h
        b[4 * i + 1] = nodes_y[i + 1]
        if i > 0:
            h = nodes_x[i] - nodes_x[i - 1]
            # S'i-1(xi) = S'i(xi)
            A[4 * i + 2][4 * (i-1) + 1] = 1
            A[4 * i + 2][4 * (i-1) + 2] = 2 * h
            A[4 * i + 2][4 * (i-1) + 3] = 3 * h * h
            A[4 * i + 2][4 * i + 1] = -1
            # S''i-1(xi) = S''i(xi)
            A[4 * i + 3][4 * (i - 1) + 2] = 2
            A[4 * i + 3][4 * (i - 1) + 3] = 6 * h
            A[4 * i + 3][4 * i + 2] = -2
    # Na krawędziach
    # S''0(x0) = 0
    A[2][2] = 2
    # S''n-1(xn) = 0
    h = nodes_x[n-1] - nodes_x[n-2]
    A[3][4 * (n - 1 - 1) + 2] = 2
    A[3][4 * (n - 1 - 1) + 3] = 6 * h

    factors = factorization_LU_with_pivoting(A, x, b)
    interpolated_values = []
    interpolated_x = []
    for xp in input_x:
        if nodes_x[0] <= xp <= nodes_x[-1]:
            interpolated_values.append(spline_interpolate_point(nodes_x, factors, xp))
            interpolated_x.append(xp)
    return interpolated_x, interpolated_values


def print_plot(input_x, input_y, nodes_x, nodes_y, interpolated_x, interpolated_y, interpolation_type, file_name):
    plt.yscale('log')
    plt.plot(input_x, input_y, 'o', markersize=2, label='Próbki')
    plt.plot(interpolated_x, interpolated_y, label='Funkcja interpolowana')
    plt.plot(nodes_x, nodes_y, 'o', label='Węzły interpolacji')
    plt.title('(' + file_name + ') ' + interpolation_type + ' dla ' + str(len(nodes_x)) + ' węzłów')
    plt.xlabel('Dystans [m]')
    plt.ylabel('Wysokość [m n.p.m.]')
    plt.legend()
    save_name = file_name + str(len(nodes_x))
    if interpolation_type == 'Interpolacja Lagrange':
        save_name += '_lagrange'
    elif interpolation_type == 'Interpolacja Splajnami':
        save_name += '_spline'
    else:
        save_name += '_lagrange_n'
    plt.savefig('plots/' + save_name + '.png')
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Przygotowanie danych
    file_name = 'MountEverest'
    #  df = pd.read_csv('input_data/' + file_name + '.txt', sep=' ')  # dla chelm.txt
    #  df = pd.read_csv('input_data/' + file_name + '.txt')  # dla chelm.txt
    df = pd.read_csv('input_data/' + file_name + '.csv')
    samplesSize = len(df.index)
    nodes_Amounts = [4, 7, 9, 15, 20, 45]
    for nodesAmount in nodes_Amounts:
        subintervals = nodesAmount - 1
        step = 1  # Tworzenie równych podprzedziałów
        while subintervals * (step + 1) < samplesSize:
            step += 1
        first_node = 0
        # wyśrodkowywanie punktów
        while (first_node + 1) < (samplesSize - 1 - (first_node + subintervals*step)):
            first_node += 1

        nodesIndexes = [first_node]
        for i in range(1, nodesAmount):  # Wyznaczone węzły interpolacji
            nodesIndexes.append(first_node + i * step)
        print(nodesIndexes)

        inputDistanceX = df[df.columns[0]].tolist()
        inputHeightY = df[df.columns[1]].tolist()
        nodesHeightY = []  # Lista węzłów do interpolacji
        nodesDistanceX = []
        for index in nodesIndexes:
            nodesHeightY.append(inputHeightY[index])
            nodesDistanceX.append(inputDistanceX[index])
        #print(nodesHeightY)

        interpolated_values_lagrange = lagrange(inputDistanceX, nodesDistanceX, nodesHeightY)
        x_for_interpolation, interpolated_values_spline = spline(inputDistanceX, nodesDistanceX, nodesHeightY)
        #  plt.plot([i for i in range(0, samplesSize)], inputPoints, label='Próbki')

        print_plot(inputDistanceX, inputHeightY, nodesDistanceX, nodesHeightY,
                   inputDistanceX, interpolated_values_lagrange, 'Interpolacja Lagrange', file_name)
        print_plot(inputDistanceX, inputHeightY, nodesDistanceX, nodesHeightY,
                   x_for_interpolation, interpolated_values_spline, 'Interpolacja Splajnami', file_name)
        if nodesAmount == 20:  # nierównomiernie rozmieszczone punkty:
            indexes = [0, 20, 40, 60, 80, 100, 150, 200, 30, 250, -1, -20, -40, -60, -80, -100, -150, -200, -30, -250]
            nodesHeightYn = []  # Lista węzłów do interpolacji
            nodesDistanceXn = []
            for i in indexes:
                nodesDistanceXn.append(inputDistanceX[i])
                nodesHeightYn.append(inputHeightY[i])
            interpolated_values_lagrange_n = lagrange(inputDistanceX, nodesDistanceXn, nodesHeightYn)
            print_plot(inputDistanceX, inputHeightY, nodesDistanceXn, nodesHeightYn,
                       inputDistanceX, interpolated_values_lagrange_n, 'Interpolacja Lagrange (Nierównomierne rozstawienie)', file_name)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
