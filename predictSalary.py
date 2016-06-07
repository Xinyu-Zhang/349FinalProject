import csv
import collections
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neural_network import BernoulliRBM
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def parse(filename):
    data = []
    label = []
    csvfile = open(filename, 'rb')
    fileToRead = csv.reader(csvfile, delimiter=',', quotechar=',')

    # skip first line of data
    fileToRead.next()

    options = {"C": -10, "SG": -5, "SF": 0, "PG": 5, "PF": 10}

    # iterate through rows of actual data
    for row in fileToRead:
        # change each line of data into an array
        # temp = row[0].split(',')
        eachrow = []
        salary = (((int(row[1])*100)/int(row[18]))/5)*5
        if salary > 25:
            salary = 25
        if salary < 5:
            salary = 5
        label.append(salary)
        i = 2
        while i < 17:
            # data preprocessing
            eachrow.append(float(row[i]))
            i += 1
        eachrow.append(options[str(row[17])])
        # rotate data so that the target attribute is at index 0
        d = collections.deque(eachrow)
        data.append(list(d))

    # array.pop()
    return data, label


def compareClf(trainData, trainTarget):
    clf_RF = RandomForestClassifier(n_estimators=100)
    scores_RF = cross_validation.cross_val_score(clf_RF, trainData, trainTarget, cv=10)
    print ("Accuracy of RF: %0.2f " % scores_RF.mean())

    clf_NB = BernoulliNB(alpha=0.5)
    scores_NB = cross_validation.cross_val_score(clf_NB, trainData, trainTarget, cv=10)
    print ("Accuracy of NB: %0.2f " % scores_NB.mean())

    clf_SVM = SVC(C=0.5, cache_size=500, class_weight=None, coef0=0.0, decision_function_shape=None, degree=5,
                  gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
    scores_SVM = cross_validation.cross_val_score(clf_SVM, trainData, trainTarget, cv=10)
    print ("Accuracy of SVM: %0.2f " % scores_SVM.mean())

    clf_KNN = KNeighborsClassifier(n_neighbors=20)
    scores_KNN = cross_validation.cross_val_score(clf_KNN, trainData, trainTarget, cv=10)
    print ("Accuracy of KNN: %0.2f " % scores_KNN.mean())

    # clf_AB = AdaBoostClassifier(base_estimator=clf_NB, learning_rate=0.1, algorithm="SAMME.R", n_estimators=100)
    # scores_AB = cross_validation.cross_val_score(clf_AB, trainData, trainTarget, cv=10)
    # print ("Accuracy of AB: %0.2f " % scores_AB.mean())


def plot_confusion_matrix(cm, title, cmap=plt.cm.Blues):
    target_names = ["5%", "10%", "15%", "20%", "25%"]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def confusionMatrix(trainData, trainTarget, testData, testTarget):
    clf_RF = RandomForestClassifier(n_estimators=300)
    clf_NB = BernoulliNB(alpha=0.5)
    clf_SVM = SVC(C=0.5, cache_size=500, class_weight=None, coef0=0.0, decision_function_shape=None, degree=5,
                  gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
    clf_KNN = KNeighborsClassifier(n_neighbors=20)
    labels = [5, 10, 15, 20, 25]
    np.set_printoptions(precision=2)
    pre_RF = clf_RF.fit(trainData, trainTarget).predict(testData)
    cm_RF = confusion_matrix(testTarget, pre_RF, labels=labels)
    cm_RF = cm_RF.astype('float') / cm_RF.sum(axis=1)[:, np.newaxis]
    pre_NB = clf_NB.fit(trainData, trainTarget).predict(testData)
    cm_NB = confusion_matrix(testTarget, pre_NB, labels=labels)
    cm_NB = cm_NB.astype('float') / cm_NB.sum(axis=1)[:, np.newaxis]
    pre_SVM = clf_SVM.fit(trainData, trainTarget).predict(testData)
    cm_SVM = confusion_matrix(testTarget, pre_SVM, labels=labels)
    cm_SVM = cm_SVM.astype('float') / cm_SVM.sum(axis=1)[:, np.newaxis]
    pre_KNN = clf_KNN.fit(trainData, trainTarget).predict(testData)
    cm_KNN = confusion_matrix(testTarget, pre_KNN, labels=labels)
    cm_KNN = cm_KNN.astype('float') / cm_KNN.sum(axis=1)[:, np.newaxis]
    plt.figure(1)
    plot_confusion_matrix(cm_RF, "Confusion Matrix of RF")
    plt.figure(2)
    plot_confusion_matrix(cm_NB, "Confusion Matrix of NB")
    plt.figure(3)
    plot_confusion_matrix(cm_SVM, "Confusion Matrix of SVM")
    plt.figure(4)
    plot_confusion_matrix(cm_KNN, "Confusion Matrix of KNN")


def featureImportance(trainData, trainTarget, testData, testTarget):
    clf_RF = RandomForestClassifier(n_estimators=100)
    y_pred = clf_RF.fit(train_data, train_label).predict(test_data)
    print accuracy_score(test_label, y_pred)
    importances = clf_RF.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf_RF.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    plt.figure(5)
    plt.title("Feature importances")
    plt.bar(range(train_data.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    options = {0: "Age", 1: "GP", 2: "GS", 3: "MIN", 4: "FG%", 5: "3P%", 6: "FT%", 7: "OR", 8: "DR", 9: "AST",
               10: "STL", 11: "BLK", 12: "TO", 13: "PF", 14: "PTS", 15: "Position"}
    attributes = []
    for i in indices:
        attributes.append(options[i])
    plt.xticks(range(train_data.shape[1]), attributes, rotation=60)
    plt.xlim([-1, train_data.shape[1]])


def errorRate(train_data, train_label, test_data, test_label):
    n_estimators = np.concatenate(([1], range(10, 500, 10)), axis=0)
    error = []
    for n in n_estimators:
        clf_RF = RandomForestClassifier(n_estimators=n)
        error.append(1 - accuracy_score(test_label, clf_RF.fit(train_data, train_label).predict(test_data)))
    plt.figure(6)
    plt.plot(n_estimators, error)
    plt.ylim((0.0, 1.0))
    plt.xlabel('n_estimators')
    plt.ylabel('error rate')
    plt.title("Error rate of RF algorithm with different number of estimators")


train_data = []
train_label = []

train_data, train_label = parse("./DATA_processed/2012.csv")
data, label = parse("./DATA_processed/2013.csv")
train_data = np.concatenate((train_data, data), axis=0)
train_label = np.concatenate((train_label, label), axis=0)
data, label = parse("./DATA_processed/2014.csv")
train_data = np.concatenate((train_data, data), axis=0)
train_label = np.concatenate((train_label, label), axis=0)
data, label = parse("./DATA_processed/2015.csv")
train_data = np.concatenate((train_data, data), axis=0)
train_label = np.concatenate((train_label, label), axis=0)

compareClf(train_data, train_label)

test_data, test_label = parse("./DATA_processed/2016.csv")
confusionMatrix(train_data, train_label, test_data, test_label)
featureImportance(train_data, train_label, test_data, test_label)
errorRate(train_data, train_label, test_data, test_label)
plt.show()

