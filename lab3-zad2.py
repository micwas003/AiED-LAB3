# Algorytmy i Eksploracja Danych - Laboratorium 3 - Zadanie 1 i 2
import pandas as pd

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# Zadanie 1 - ladowanie danych
def ReadDataSet():
    global Y_train, X_train, Y_test, X_test
    Y_train = pd.read_csv("dataset/train/y_train.txt", sep=' ', header=None)
    X_train = pd.read_csv("dataset/train/X_train.txt", sep=' ', header=None)
    Y_test = pd.read_csv("dataset/test/y_test.txt", sep=' ', header=None)
    X_test = pd.read_csv("dataset/test/X_test.txt", sep=' ', header=None)

    #print('Dataset Y train: ')
    #print(Y_train)
    #print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.')
    #print('Dataset X train: ')
    #print(X_train)
    #print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.')
    #print('Dataset Y test: ')
    #print(Y_test)
    #print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.')
    #print('Dataset X test: ')
    #print(X_test)


# Zadanie 2 - Tworzenie modeli SVM, KNN, Decision Tree, Random Forest
def CreateModelsClassification():
    global models, models_names, models_predict, models_predict_proba
    print(type(X_train), type(Y_train))
    print(X_train)
    print(Y_train.values.ravel())

    # Classification - SVM
    print('Classification - SVM')
    classificationSVM = svm.SVC(probability=True)
    classificationSVM.fit(X_train, Y_train.values.ravel())
    predictSVM = classificationSVM.predict(X_test)
    predictprobaSVM = classificationSVM.predict_proba(X_test)
    print("SVM predict results:")
    print(predictSVM)


    # Classification - KNN
    print('Classification - KNN')
    classificationKNN = KNeighborsClassifier()
    classificationKNN.fit(X_train, Y_train.values.ravel())
    predictKNN = classificationKNN.predict(X_test)
    predictprobaKNN = classificationKNN.predict_proba(X_test)
    print("KNN predict results:")
    print(predictKNN)

    # Classification - Decision Tree
    print('Classification - Decision Tree')
    classificationDecisionTree = tree.DecisionTreeClassifier()
    classificationDecisionTree = classificationDecisionTree.fit(X_train, Y_train.values.ravel())
    predictDecisionTree = classificationDecisionTree.predict(X_test)
    predictprobaDecisionTree = classificationDecisionTree.predict_proba(X_test)
    print("Decision Tree predict result:")
    print(predictDecisionTree)

    # Classification - Random Forest
    print('Classification - Random Forest')
    classificationRandomForest = RandomForestClassifier()
    classificationRandomForest.fit(X_train, Y_train.values.ravel())
    predictRandomForest = classificationRandomForest.predict(X_test)
    predictprobaRandomForest = classificationRandomForest.predict_proba(X_test)
    print("Random Forest predict result:")
    print(predictRandomForest)
    print(type(classificationSVM), type(classificationKNN))

    models_names = ['Classification - SVM', 'Classification - KNN', 'Classification - Decision Tree', 'Classification - Random Forest']
    models = [classificationSVM, classificationKNN, classificationDecisionTree, classificationRandomForest]
    models_predict = [predictSVM, predictKNN, predictDecisionTree, predictRandomForest]
    models_predict_proba = [predictprobaSVM, predictprobaKNN, predictprobaDecisionTree, predictprobaRandomForest]

if __name__ == "__main__":
    ReadDataSet()
    CreateModelsClassification()
