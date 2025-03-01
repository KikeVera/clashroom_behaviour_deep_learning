import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import StackingClassifier
import matplotlib.pyplot as plt

# 'hidden_layer_sizes': (1000,)
class MachineLearningUtils:
    def __init__(self):
        self.hpknn = {'n_neighbors': 4, 'algorithm': 'auto', 'weights': 'distance', 'p': 1, 'leaf_size': 15}
        self.hpdt = {'criterion': "log_loss", 'min_samples_split': 2, 'min_samples_leaf': 1, 'ccp_alpha': 0.001,
                     'class_weight': 'balanced',
                     'max_features': 'sqrt', 'min_impurity_decrease': 1e-4}
        self.hpnw = {'max_iter': 50000, 'activation': 'logistic', 'learning_rate': 'adaptive',
                     'shuffle': False, 'batch_size': 256, 'tol': 1e-4, 'beta_1': 0.8, 'beta_2': 0.9999,
                     'n_iter_no_change': 50, 'alpha': 0.00008, 'hidden_layer_sizes': (1000,)}
        self.hprf = {'n_jobs': -1, 'criterion': "entropy", 'n_estimators': 100,
                     'bootstrap': False}
        self.hpxt = {'n_jobs': -1, 'criterion': "entropy", 'n_estimators': 100,
                     'bootstrap': False}

    def get_training_data(self, train_df: pd.DataFrame, predicted_var, test_size=0.2):
        x = train_df.drop(predicted_var, axis=1)
        y = train_df[predicted_var]

        # Split the data into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        return x_train, x_test, y_train, y_test

    def get_external_test(self, train_df: pd.DataFrame, predicted_var, test_size=0.999):
        x = train_df.drop(predicted_var, axis=1)
        y = train_df[predicted_var]

        # Split the data into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        return x_test, y_test

    def search_best_knn(self, x_train, y_train):
        knn = KNeighborsClassifier()
        parameters = dict(weights=['uniform', 'distance'], algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'],
                          n_neighbors=list(range(3, 5)))
        clf = GridSearchCV(knn, parameters)
        clf.fit(x_train, y_train)
        print(clf.best_score_)
        print(clf.best_params_)

        knn = KNeighborsClassifier(algorithm=clf.best_params_['algorithm'], weights=clf.best_params_['weights'],
                                   n_neighbors=clf.best_params_['n_neighbors'], p=1, leaf_size=15)
        parameters = dict()

        clf = GridSearchCV(knn, parameters)
        clf.fit(x_train, y_train)
        print(clf.best_score_)
        print(clf.best_params_)

    def search_best_neuronal_network(self, x_train, y_train):
        clf = MLPClassifier(max_iter=50000, activation='logistic', learning_rate='adaptive',
                            shuffle=False, batch_size=256, tol=1e-4, beta_1=0.8,
                            beta_2=0.9999, n_iter_no_change=100, epsilon=1e-07,
                            alpha=0.00008, hidden_layer_sizes=(1000,))
        parameters = dict(hidden_layer_sizes=[(800,), (1000,), (1200,)])
        clf = GridSearchCV(clf, parameters)
        clf.fit(x_train, y_train)
        print(clf.best_score_)
        print(clf.best_params_)
        print(clf.cv_results_)

    def search_best_svm(self, x_train, y_train):
        svm = SVC()
        parameters = dict(kernel=['linear', 'poly', 'rbf', 'sigmoid'])
        svm = GridSearchCV(svm, parameters)
        svm.fit(x_train, y_train)
        print(svm.best_score_)
        print(svm.best_params_)
        print(svm.cv_results_)

    def search_best_random_forest(self, x_train, y_train):
        rf = RandomForestClassifier(n_jobs=-1, criterion="entropy", max_features='auto', n_estimators=500,
                                    bootstrap=False)
        parameters = dict()
        rf = GridSearchCV(rf, parameters)
        rf.fit(x_train, y_train)
        print(rf.best_score_)
        print(rf.best_params_)
        print(rf.cv_results_)

    def knn_train(self, x_train, x_test, y_train, y_test, hp=None):
        if hp is None:
            hp = self.hpknn
        knn = KNeighborsClassifier(**hp)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy knn:", accuracy)

        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 knn:", f1)

        return knn

    def decision_tree_train(self, x_train, x_test, y_train, y_test, hp=None):
        if hp is None:
            hp = self.hpdt
        tree = DecisionTreeClassifier(**hp)
        tree.fit(x_train, y_train)

        y_pred = tree.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy tree:", accuracy)

        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 tree:", f1)

        return tree

    def svm_train(self, x_train, x_test, y_train, y_test, hp=None):
        if hp is None:
            hp = {}
        svm = SVC(**hp)
        svm.fit(x_train, y_train)

        y_pred = svm.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy svm:", accuracy)

        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 svm:", f1)
        return svm

    def neuronal_network_train(self, x_train, x_test, y_train, y_test, hp=None):
        if hp is None:
            hp = self.hpnw
        clf = MLPClassifier(**hp)
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy neuronal network:", accuracy)

        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 neuronal network:", f1)

        return clf

    def bayes_train(self, x_train, x_test, y_train, y_test, hp=None):
        if hp is None:
            hp = {}
        gnb = GaussianNB(**hp)
        gnb.fit(x_train, y_train)

        y_pred = gnb.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy bayes:", accuracy)

        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 gnb:", f1)

        return gnb

    def random_forest_train(self, x_train, x_test, y_train, y_test, hp=None):
        if hp is None:
            hp = self.hprf
        rf = RandomForestClassifier(**hp)
        rf.fit(x_train, y_train)

        y_pred = rf.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy random forest:", accuracy)

        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 random forest:", f1)

        return rf

    def extra_tree_train(self, x_train, x_test, y_train, y_test, hp=None):
        if hp is None:
            hp = self.hprf
        rf = ExtraTreesClassifier(**hp)
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy extra tree:", accuracy)

        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 extratree:", f1)

        return rf

    def stochastic_gradient_train(self, x_train, x_test, y_train, y_test, hp=None):
        if hp is None:
            hp = {}
        sg = SGDClassifier(**hp)
        sg.fit(x_train, y_train)

        y_pred = sg.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy stochastic gradient:", accuracy)

        return sg

    def stacking_train(self, x_train, x_test, y_train, y_test, knn=True, mlp=True, rf=True, xtree=True):
        estimators = []
        model_name = ""
        if knn:
            estimators.append(
                ("knn", KNeighborsClassifier(n_neighbors=4, algorithm='auto', weights='distance', p=1, leaf_size=15)))
            model_name += "knn_"
        if mlp:
            estimators.append(
                ("mlp", MLPClassifier(max_iter=5000000, activation='logistic', learning_rate='adaptive',
                                      shuffle=False, batch_size=256, tol=1e-4, beta_1=0.8, beta_2=0.9999,
                                      n_iter_no_change=50, hidden_layer_sizes=(1000,), alpha=0.00008)))
            model_name += "mlp_"
        if rf:
            estimators.append(
                ("rf",
                 RandomForestClassifier(n_jobs=-1, criterion="entropy", n_estimators=300,
                                        bootstrap=False)))
            model_name += "rf_"
        if xtree:
            estimators.append(
                ("xtree", ExtraTreesClassifier(n_jobs=-1, criterion="entropy", n_estimators=300,
                                               bootstrap=False)))
            model_name += "xtree_"

        model_name += "model"
        rf = StackingClassifier(estimators=estimators)
        rf.fit(x_train, y_train)
        # rf = pickle.load(open("./data/models/stacking_model.pkl", 'rb'))
        y_pred = rf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy ", model_name, ": ", accuracy)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print("F1 ", model_name, ": ", f1)

        return rf

    def plot_confusion_matrix(self, x_test, y_test, model, model_name):
        np.set_printoptions(precision=2)
        disp = ConfusionMatrixDisplay.from_estimator(
            model,
            x_test,
            y_test,
            cmap=plt.cm.Blues,
            normalize='true',
            xticks_rotation=45

        )
        disp.ax_.set_title('Confusion matrix ' + model_name)

    def check_accuracy_of_real_users(self, model, real_df, predicted_df, user, repredict = True):
        x_real = real_df.drop("action", axis=1)
        y_real = real_df["action"]
        y_predicted = predicted_df["action"]
        if repredict:
            y_predicted=model.predict(predicted_df.drop("action", axis=1)).transpose()

        accuracy = accuracy_score(y_real, y_predicted)
        print("Accuracy:", accuracy)
        # self.plot_confusion_matrix(x_test=x_real, y_test=y_real, model=model, model_name=user)
        # plt.show()
