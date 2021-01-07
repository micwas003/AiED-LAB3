# Algorytmy i Eksploracja Danych - Laboratorium 3 - Zadanie 1
import pandas as pd

# Zadanie 1 - ladowanie danych
def ReadDataSet():
    global Y_train, X_train, Y_test, X_test
    Y_train = pd.read_csv("dataset/train/y_train.txt", sep=' ', header=None)
    X_train = pd.read_csv("dataset/train/X_train.txt", sep=' ', header=None)
    Y_test = pd.read_csv("dataset/test/y_test.txt", sep=' ', header=None)
    X_test = pd.read_csv("dataset/test/X_test.txt", sep=' ', header=None)

    print('Dataset Y train: ')
    print(Y_train)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.')
    print('Dataset X train: ')
    print(X_train)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.')
    print('Dataset Y test: ')
    print(Y_test)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.')
    print('Dataset X test: ')
    print(X_test)

if __name__ == "__main__":
    ReadDataSet()
