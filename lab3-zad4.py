# Algorytmy i Eksploracja Danych - Laboratorium 3 - Zadanie 1,2,3 i 4
import pandas as pd

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score

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

# Zadanie 3 - Metryki Oceny Klasyfikatorow: Confusion Matrix, ACC, Recall, F1 score, AUC, Cross Validation
def MetricsClassification():
    # Metric Classification - Confusion matrix
    print('Metric Classification - Confusion matrix')
    for i in range(len(models_predict)):
        print('Confusion matrix metric for model:', models_names[i])
        print(confusion_matrix(Y_test.values.ravel(), models_predict[i]))

    # Metric Classification - Accuracy Score
    print('Metric Classification - Accuracy Score')
    for i in range(len(models_predict)):
        acc = accuracy_score(Y_test.values.ravel(), models_predict[i])
        print('ACC metric for {} = {}'.format(models_names[i], acc))

    # Metric Classification - Recall (macro)
    print('Metric Classification - Recall (macro)')
    for i in range(len(models_predict)):
        recall_macro = recall_score(Y_test.values.ravel(), models_predict[i], average='macro')
        print('Recall (macro) metric for {} = {}'.format(models_names[i], recall_macro))

    # Metric Classification - Recall (micro)
    print('Metric Classification - Recall (micro)')
    for i in range(len(models_predict)):
        recall_micro = recall_score(Y_test.values.ravel(), models_predict[i], average='micro')
        print('Recall (micro) metric for {} = {}'.format(models_names[i], recall_micro))

    # Metric Classification - F1 (macro)
    print('Metric Classification - F1 (macro)')
    for i in range(len(models_predict)):
        f1_macro = f1_score(Y_test.values.ravel(), models_predict[i], average='macro')
        print('F1 (macro) metric for {} = {}'.format(models_names[i], f1_macro))

    # Metric Classification - F1 (micro)
    print('Metric Classification - F1 (micro)')
    for i in range(len(models_predict)):
        f1_micro = f1_score(Y_test.values.ravel(), models_predict[i], average='micro')
        print('F1 (micro) metric for {} = {}'.format(models_names[i], f1_micro))

    # Metric Classification - AUC (macro)
    print('Metric Classification - AUC (macro)')
    for i in range(len(models_predict)):
        auc_macro = roc_auc_score(Y_test.values.ravel(), models_predict_proba[i], average='macro', multi_class='ovr')
        print('AUC (macro) metric for {} = {}'.format(models_names[i], auc_macro))

    # Metric Classification - AUC (weighted)
    print('Metric Classification - AUC (weighted)')
    for i in range(len(models_predict)):
        auc_micro = roc_auc_score(Y_test.values.ravel(), models_predict_proba[i], average='weighted', multi_class='ovr')
        print('AUC (weighted) metric for {} = {}'.format(models_names[i], auc_micro))

    # Metric Classification - Cross Validation
    print('Metric Classification - Cross Validation')
    for i in range(len(models)):
        print(models_names[i])
        score = cross_val_score(models[i], X_train, Y_train.values.ravel(), cv=5)
        print('Cross Validation - Average:', score.mean())
        print('Cross Validation - Standard deviation :', score.std())

# Zadanie 4 - Optymalne wartosci parametrow klasyfikacji
def OptimizeClasifiction():

    # SVM - Evaluate score by cross-validation
    print('SVM - Evaluate score by cross-validation')
    n_value = 0
    top_score = 0
    for i in range(1, 8):
        classifictionSVM = svm.SVC(C=i/8, probability=True)
        classifictionSVM.fit(X_train, Y_train.values.ravel())
        top_value = cross_val_score(classifictionSVM, X_test, Y_test.values.ravel(), cv=4)
        if top_value.mean() > top_score:
            top_score = top_value.mean()
            n_value = i / 8
    print('Najbardziej znaczacy wynik: {} otrzymano dla: {}'.format(top_score, n_value))

    # KNN - Evaluate score by cross-validation
    print('KNN - Evaluate score by cross-validation')
    n_neighbors = 0
    top_score = 0
    for i in range(1, 8):
        classificationKNN = KNeighborsClassifier(n_neighbors=i)
        classificationKNN.fit(X_train, Y_train.values.ravel())
        top_value = cross_val_score(classificationKNN, X_test, Y_test.values.ravel(), cv=4)
        if top_value.mean() > top_score:
            top_score = top_value.mean()
            n_neighbors = i
    print('Najbardziej znaczacy wynik: {} otrzymano dla: {}'.format(top_score, n_neighbors))


if __name__ == "__main__":
    ReadDataSet()
    CreateModelsClassification()
    MetricsClassification()
    OptimizeClasifiction()
