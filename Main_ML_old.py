#!/usr/bin/en# subject-dependent
# nested K-fold CV
import warnings
#import seaborn as sns
import matplotlib
matplotlib.use('TKAgg')

from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA, PCA
from sklearn import decomposition
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model, metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression,Ridge,Lasso
from sklearn.preprocessing import Binarizer
from itertools import product
from numpy import unravel_index
from Classes.Path import Path_info
import matplotlib.font_manager as font_manager
from sklearn.linear_model import LinearRegression
from Classes.Data import Load_data, Create_DB, Up_sampling
#from tensorflow.python.client import device_lib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel,ConstantKernel, RBF
import time
from scipy.stats import mode
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
import time
from sklearn import tree, svm, neighbors, ensemble
import scipy.stats
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def compute_balanced_accuracy(true_class, predict_class):
    true_class_dict = {}
    accuracy = {}
    true_class=true_class.squeeze()
    predict_class=predict_class.squeeze()

    for key in true_class:
        true_class_dict[key] = true_class_dict.get(key, 0) + 1

        i = np.argwhere(true_class == key)
        true_class_key = np.array([true_class[x] for x in i])
        predict_class_key = np.array([predict_class[x] for x in i])

        n_right = len(np.where(predict_class_key == true_class_key)[0])
        n_total = len(true_class_key)
        accuracy[key] = n_right / n_total

    accuracy_mean = np.mean([accuracy[key] for key in accuracy])
    return accuracy_mean


def compute_accuracy(z_train, z_test, y_train, y_test, nn=10):
    neigh = KNeighborsClassifier(n_neighbors=nn)
    neigh.fit(z_train, y_train)

    predicted_ind = neigh.predict(z_test)

    accuracy = np.round(metrics.accuracy_score(y_test, predicted_ind), decimals=4)
    accuracy = np.round(compute_balanced_accuracy(y_test, predicted_ind), decimals=4)

    return accuracy, predicted_ind


def plot(dim_reduction, model_name, df, DB_N):
    acc_valid = df.loc[model_name]['Valid accuracy']
    acc_test = str(df.loc[model_name]['Test accuracy'])
    knn_valid = str(df.loc[model_name]['Best knn'])

    load_data = Load_data(DB_N, which_model='MLP')
    X_train, X_valid, Y_train, Y_valid, Y_orig_train, Y_orig_valid = load_data.K_fold_data(K=0)
    X_test, Y_test, Y_orig_test = load_data.test_data()

    unique_ind_train = np.unique(Y_train)
    unique_ind_test = np.unique(Y_test)

    cmap = plt.get_cmap('nipy_spectral')
    colors = cmap(np.linspace(0, 1, len(unique_ind_train)))
    colors = dict(zip(unique_ind_train, colors))

    fig = plt.figure(figsize=(16, 10))

    ax = fig.add_subplot(1, 1, 1)

    for j in unique_ind_train:
        label = j

        indices = [Y_train == j]

        x = dim_reduction.results_train[model_name][indices[0], 0]
        y = dim_reduction.results_train[model_name][indices[0], 1]
        l = np.unique(np.array(Y_train)[indices[0]])

        ax.scatter(x, y, color=colors[j], s=80, alpha=0.4, edgecolor='black', linewidth='0.1')

        indices = [Y_test == j]

        x = dim_reduction.results_test[model_name][indices[0], 0]
        y = dim_reduction.results_test[model_name][indices[0], 1]

        ax.scatter(x, y, color=colors[j], s=150, marker="X", edgecolor='black', linewidth='1', label=label)

        ax.set_title(model_name)

    fig.subplots_adjust(left=0.2, bottom=0.05, right=0.8, top=0.9, wspace=0.1, hspace=0.25)

    font = font_manager.FontProperties(family='Comic Sans MS', style='normal', size=14)
    ax.legend(prop=font, loc='upper left', bbox_to_anchor=(1, 1))

    ax.set_title(
        model_name + " \n \n  Valid accuracy: " + str(acc_valid) + " - Test accuracy: " + acc_test + " with knn=" + str(
            knn_valid))

    return fig


def ranked_accuracy(df_sup_class, df_unsup):
    max_df = df_sup_class.sort_values(by='accuracy')
    for index, row in max_df.iterrows():
        print("Ranked classifier: " + row["method"] + " with an accuracy of " + str(row["accuracy"]))

    print("\n")
    print("\n")

    max_df = df_unsup.sort_values(by='accuracy')
    for index, row in max_df.iterrows():
        print(
            "Ranked clustering: " + row["method"] + " with an accuracy of " + str(row["accuracy"]) + " for knn=" + str(
                row["knn"]))


def get_hyperparameters(model_name):
    # Set the parameters by cross-validation
    # Clsutering
    if model_name == 'PCA':
        tuned_parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']}
    elif model_name == 'TSNE':
        tuned_parameters = {'perplexity': [5, 10, 20, 50, 100], 'learning_rate': [100]}
    elif model_name == 'ICA':
        tuned_parameters = {'algorithm': ['parallel', 'deflation']}
    elif model_name == 'MDS':
        tuned_parameters = {'metric': [True, False]}
    elif model_name == 'ISO':
        tuned_parameters = {'n_neighbors': [5, 10, 20]}
    elif model_name == 'LLE':
        tuned_parameters = {'n_neighbors': [5, 10, 20]}
    elif model_name == 'laplacian':
        tuned_parameters = {'affinity': ['nearest_neighbors', 'rbf'], 'n_neighbors': [5, 10, 20]}
    elif model_name == 'LDA':
        tuned_parameters = {'solver': ['svd']}

    # Classification
    elif model_name == 'SVM':
        tuned_parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    elif model_name == 'LDA':
        tuned_parameters = {'solver': ['svd']}
    elif model_name == 'QDA':
        tuned_parameters = {'nothing': ['nothing']}
    elif model_name == 'RF':
        tuned_parameters = {'n_estimators': [10, 500, 1000], 'max_depth': [11,2,3]}
    elif model_name == 'RF_Regression':
        tuned_parameters = {'n_estimators': [10,100,500], 'max_depth': [11,3,2],'max_features': ['auto'],
                            'min_samples_split': [2,1], 'min_samples_leaf': [2,1],'bootstrap': [True, False]}

    elif model_name == 'GP':
        tuned_parameters = {'kernel': ['DotProduct']}

    elif model_name == 'MLP':
        tuned_parameters = {'hidden_layer_sizes': [(64), (64, 128), (64, 128, 256), (32), (32, 64), (32, 64, 128)],
                            'activation': ['tanh', 'relu', 'logistic'], 'solver': ['adam', 'lbfgs', 'sgd']}
    elif model_name == 'SVC':
        tuned_parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    elif model_name == 'KNN_Classifier': # K-neighbours nearest
        tuned_parameters = {'n_neighbors': [5, 10, 25,30]}
    elif model_name == "LR":
        tuned_parameters = {'solver': ['newton-cg', 'sag', 'saga', 'lbfgs'],'penalty': ['l2'], 'C': [0.001,0.1,1,10],'multi_class':['ovr', 'multinomial']}
    elif model_name == "RBM":
        tuned_parameters = {'learning_rate': [0.01, 0.005, 0.001]}
    elif model_name == "Ridge":
        tuned_parameters = {'alpha': [0.01, 0.005, 0.001,0.1,1,10,100,1000],'Fit_intercept':[True,False],'Normalize':[True,False]}
    elif model_name == "Lasso":
        tuned_parameters = {'alpha': [0.01, 0.005, 1]}

    elif model_name == 'DecisionTreeClassifier':
        tuned_parameters = {'max_depth': [2, 10, 32]}

    elif model_name == 'GradientBoostingClassifier':  # Random forest
        tuned_parameters = {'n_estimators': [100, 50, 10]}

    elif model_name == 'AdaBoostClassifier':  # Random forest
        tuned_parameters = {'n_estimators': [100, 50, 10]}


    keys = tuned_parameters.keys()
    hyperparameters = []
    for vals in product(*tuned_parameters.values()):
        hyperparameters.append(dict(zip(tuned_parameters, vals)))

    return hyperparameters


class Machine_Learning_clustering():

    def __init__(self, DB_N, subject):

        self.results = {}
        self.results_train = {}
        self.results_test = {}

        self.results_classification = {}
        self.DB_N = DB_N
        self.subject = subject

        self.nns = [i for i in range(1, 100)]

        self.load_data = Load_data(DB_N=self.DB_N, which_model='MLP', subject=self.subject)

    def get_model(self, model_name, hyperparameter):

        if model_name == 'PCA':

            kernel = hyperparameter['kernel']
            print("Performing PCA " + kernel + " - K " + str(self.current_K))
            model = decomposition.KernelPCA(n_components=self.n_components, kernel=kernel)

            name = 'PCA_' + kernel

        elif model_name == 'TSNE':

            learning_rate = hyperparameter['learning_rate']
            perplexity = hyperparameter['perplexity']

            print("Performing TSNE perplexity: " + str(perplexity) + " learning_rate: " + str(
                learning_rate) + " - K " + str(self.current_K))

            model = TSNE(n_components=self.n_components, perplexity=perplexity, learning_rate=learning_rate,
                         random_state=0,
                         n_iter=300, verbose=1)
            name = 'TSNE_p_' + str(perplexity) + "_lr_" + str(learning_rate)

        elif model_name == 'ICA':

            algorithm = hyperparameter['algorithm']
            print("Performing ICA algorithm: " + algorithm + " - K " + str(self.current_K))
            model = FastICA(n_components=self.n_components, algorithm=algorithm)

            name = 'ICA_' + algorithm

        elif model_name == 'MDS':

            metric = hyperparameter['metric']
            print("Performing MDS metric: " + str(metric) + " - K " + str(self.current_K))
            model = MDS(n_components=self.n_components, metric=metric)

            name = 'MDS_m_' + str(metric)

        elif model_name == 'ISO':

            n_neighbors = hyperparameter['n_neighbors']
            print("Performing ISO n_neighbors: " + str(n_neighbors) + " - K " + str(self.current_K))
            model = Isomap(n_components=self.n_components, n_neighbors=n_neighbors)

            name = 'ISO_nn_' + str(n_neighbors)

        elif model_name == 'LLE':

            n_neighbors = hyperparameter['n_neighbors']
            print("Performing LLE n_neighbors: " + str(n_neighbors) + " - K " + str(self.current_K))
            model = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=self.n_components, method='modified',
                                           eigen_solver='dense')

            name = 'LLE_nn_' + str(n_neighbors)

        elif model_name == 'laplacian':

            affinity = hyperparameter['affinity']
            n_neighbors = hyperparameter['n_neighbors']

            print(
                "Performing laplacian n_neighbors: " + str(n_neighbors) + " affinity: " + str(affinity) + " - K " + str(
                    self.current_K))
            model = SpectralEmbedding(n_components=self.n_components, affinity=affinity, n_neighbors=n_neighbors)

            name = 'laplacian_' + str(affinity)

        elif model_name == 'LDA':

            solver = hyperparameter['solver']
            print("Performing LDA solver " + str(solver) + " - K " + str(self.current_K))
            model = LinearDiscriminantAnalysis(n_components=self.n_components, solver=solver)

            name = 'LDA_s_' + str(solver)



        return model, name

    def perform_clustering(self, model_names, n_components=2, use_accuracy=False):

        self.n_components = n_components

        self.results_train = {}
        self.results_test = {}

        self.y_test_true = {}
        self.y_test_pred = {}

        df = pd.DataFrame(columns=['Best knn', 'Valid accuracy', 'Test accuracy'])

        model_name = model_names[0]

        load_data = Load_data(DB_N=self.DB_N, which_model='MLP', subject=self.subject)
        # load_data = Load_data(DB_N=DB_N, which_model=which_model, subject=subject)

        self.K_fold = load_data.K_fold

        for model_name in model_names:

            hyperparameters = get_hyperparameters(model_name=model_name)

            accuracies = np.empty([len(hyperparameters), len(self.nns), self.K_fold])
            pearson_coefs = np.empty([len(hyperparameters), len(self.nns), self.K_fold])

            i = 0
            hyper = hyperparameters[i]

            K = 1

            for i, hyper in enumerate(hyperparameters):

                for K in range(self.K_fold):

                    self.current_K = K

                    X_train, X_valid, Y_train, Y_valid, Y_orig_train, Y_orig_valid = load_data.K_fold_data(
                        K=self.current_K)

                    X = np.concatenate([X_train, X_valid], axis=0)
                    Y = np.concatenate([Y_train, Y_valid], axis=0)

                    train_ind = [0, X_train.shape[0]]

                    model, name = self.get_model(model_name=model_name, hyperparameter=hyper)

                    if model_name in ['PCA', 'ICA', 'MDS', 'ISO', 'LLE']:
                        model.fit(X_train)

                        results_train = model.transform(X_train)
                        results_valid = model.transform(X_valid)

                    elif model_name in ['TSNE', 'laplacian']:

                        results = model.fit_transform(X)

                        results_train = results[train_ind[0]:train_ind[1]]
                        results_valid = results[train_ind[1]:]

                    elif model_name in ['LDA']:

                        model.fit(X_train, Y_train)

                        results_train = model.transform(X_train)
                        results_valid = model.transform(X_valid)

                    # Evaluate accuracy with KNN
                    for j, nn in enumerate(self.nns):
                        accuracy, y_valid_pred = compute_accuracy(z_train=results_train, z_test=results_valid,
                                                                  y_train=Y_train, y_test=Y_valid, nn=nn)
                        corrcoef = np.corrcoef(Y_valid, y_valid_pred)[1, 0]

                        pearson_coefs[i, j, K] = corrcoef
                        accuracies[i, j, K] = accuracy

                accuracies_mean = np.mean(accuracies, axis=2)
                accuracies_std = np.std(accuracies, axis=2)

                pearson_coefs_mean = np.mean(pearson_coefs, axis=2)
                pearson_coefs_std = np.std(pearson_coefs, axis=2)

            def info_best(accuracies_mean, accuracies_std):

                best_index_acc = unravel_index(accuracies_mean.argmax(), accuracies_mean.shape)
                best_knn_acc = self.nns[best_index_acc[1]]

                best_mean_acc = accuracies_mean[best_index_acc]
                best_std_acc = accuracies_std[best_index_acc]

                return best_index_acc, best_knn_acc, best_mean_acc, best_std_acc

            best_index_acc, best_knn_acc, best_mean_acc, best_std_acc = info_best(accuracies_mean, accuracies_std)
            best_index_pearson_coefs, best_knn_pearson_coefs, best_mean_pearson_coefs, best_std_pearson_coefs = info_best(
                pearson_coefs_mean, pearson_coefs_std)

            if use_accuracy:

                best_mean = best_mean_acc
                best_std = best_std_acc
                best_index = best_index_acc
                best_knn = best_knn_acc

            else:

                best_mean = best_mean_pearson_coefs
                best_std = best_std_pearson_coefs
                best_index = best_index_pearson_coefs
                best_knn = best_knn_pearson_coefs

            # Test
            X_train, X_valid, Y_train, Y_valid, Y_orig_train, Y_orig_valid = load_data.K_fold_data(K=0)
            X_test, Y_test, Y_orig_test = load_data.test_data()

            X = np.concatenate([X_train, X_test], axis=0)
            Y = np.concatenate([Y_train, Y_test], axis=0)

            train_ind = [0, X_train.shape[0]]

            best_hyper = hyperparameters[best_index[0]]

            model, name = self.get_model(model_name=model_name, hyperparameter=best_hyper)

            if model_name in ['PCA', 'ICA', 'MDS', 'ISO', 'LLE']:

                model.fit(X_train)

                results_train = model.transform(X_train)
                results_test = model.transform(X_test)

            elif model_name in ['TSNE', 'laplacian']:

                results = model.fit_transform(X)

                results_train = results[train_ind[0]:train_ind[1]]
                results_test = results[train_ind[1]:]

            elif model_name in ['LDA']:

                model.fit(X_train, Y_train)

                results_train = model.transform(X_train)
                results_test = model.transform(X_test)

            accuracy, y_test_pred = compute_accuracy(z_train=results_train, z_test=results_test, y_train=Y_train,
                                                     y_test=Y_test, nn=best_knn)

            if use_accuracy:

                # Accuracy
                K_acc_str = str("{0:.2f}".format(best_mean * 100)) + " (" + str("{0:.3}".format(best_std * 100)) + ")"
                Test_acc_str = str("{0:.2f}".format(accuracy * 100))

                print(str(name) + " - knn: " + str(best_knn))
                print("Valid Accuracy: " + K_acc_str)
                print("Test Accuracy: " + Test_acc_str)

                df.loc[name] = [best_knn, K_acc_str, Test_acc_str]

            else:

                # Pearson
                K_pearson_str = str("{0:.2f}".format(best_mean)) + " (" + str("{0:.3}".format(best_std)) + ")"
                Test_pearson_str = str("{0:.2f}".format(accuracy))

                print(' ')
                print(str(name) + " - knn: " + str(best_knn))
                print("Valid Pearson: " + K_pearson_str)
                print("Test Pearson: " + Test_pearson_str)

                df.loc[name] = [best_knn, K_pearson_str, Test_pearson_str]

            self.results_train[name] = results_train
            self.results_test[name] = results_test

            self.y_test_true[name] = Y_test
            self.y_test_pred[name] = y_test_pred

        return df


class Machine_Learning_classification():

    def __init__(self, DB_N, subject):

        self.results = {}
        self.results_train = {}
        self.results_test = {}

        self.results_classification = {}
        self.DB_N = DB_N
        self.subject = subject

        # self.nns = [i for i in range(1, 100)]


        self.load_data = Load_data(DB_N=self.DB_N, which_model='MLP', subject=self.subject)

    def get_model(self, model_name, hyperparameter):

        if model_name == 'SVM':

            kernel = hyperparameter['kernel']
            print("Performing SVM " + kernel + " - K " + str(self.current_K))
            model = svm.SVC(kernel=kernel, gamma='scale')

            name = 'SVM_' + kernel

        elif model_name == 'LDA':

            solver = hyperparameter['solver']
            print("Performing LDA " + solver + " - K " + str(self.current_K))
            model = LinearDiscriminantAnalysis(n_components=None, solver=solver)

            name = 'LDA_' + solver

        elif model_name == 'QDA':

            print("Performing QDA " + " - K " + str(self.current_K))
            model = QuadraticDiscriminantAnalysis()

            name = 'QDA'

        elif model_name == 'RF':

            n_estimators = hyperparameter['n_estimators']
            max_depth = hyperparameter['max_depth']

            print("Performing RF n_estimators: " + str(n_estimators) + " max_depth: " + str(max_depth)
                  + " - K " + str(self.current_K))

            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)

            name = 'RF_n_estimators_' + str(n_estimators) + "_max_depth_" + str(max_depth)

        elif model_name == 'RF_Regression':

            n_estimators = hyperparameter['n_estimators']
            max_depth = hyperparameter['max_depth']
            max_features=hyperparameter['max_features']
            min_samples_split = hyperparameter['min_samples_split']
            min_samples_leaf = hyperparameter['min_samples_leaf']
            bootstrap = hyperparameter['bootstrap']

            print("Performing RF_Regression n_estimators: " + str(n_estimators) + " max_depth: " + str(max_depth)
                  + " max_features: " + str(max_features)+ " min_samples_split: " + str(min_samples_split)+ " min_samples_leaf: " + str(min_samples_leaf)
                  + " bootstrap: " + str(bootstrap)
                  + " - K " + str(self.current_K))

            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0)

            name = 'RF_Regression_n_estimators_' + str(n_estimators) + "_max_depth_" + str(max_depth)

        elif model_name == 'GP':

            kernel = DotProduct() + WhiteKernel()
            rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
            model = GaussianProcessRegressor(kernel=kernel, alpha=0.4 ** 2)


            name = 'GP'

        elif model_name == 'MLP':

            hidden_layer_sizes = hyperparameter['hidden_layer_sizes']
            activation = hyperparameter['activation']
            solver = hyperparameter['solver']

            print("Performing MLP hidden_layer_sizes: " + str(hidden_layer_sizes) +
                  " activation: " + str(activation) + " solver: " + str(solver) + " - K " + str(self.current_K))
            model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, solver=solver, activation=activation,
                                  alpha=1e-5, random_state=1, batch_size=200, learning_rate='adaptive')

            name = 'MLP_hs_' + str(hidden_layer_sizes) + "_activation_" + str(activation) + "_solver_" + str(solver)

        elif model_name == 'SVC':

            kernel = hyperparameter['kernel']

            print("Performing SVC " + kernel + " - K " + str(self.current_K))

            model = SVC(kernel=kernel)

            name = 'SVC_' + str(kernel)

        elif model_name == 'KNN_Classifier':

            n_neighbors = hyperparameter['n_neighbors']
            print("Performing KNN n_neighbors: " + str(n_neighbors) + " - K " + str(self.current_K))

            model = KNeighborsClassifier(n_neighbors=n_neighbors)

            name = 'KNN_Classifier_nn_' + str(n_neighbors)

        elif model_name == 'LR':

            solver = hyperparameter['solver']
            penalty = hyperparameter['penalty']
            C = hyperparameter['C']
            multi_class = hyperparameter['multi_class']
            print("Performing LR solver: " + str(solver) + " - K " + str(self.current_K))

            model = LogisticRegression(random_state=0, solver=solver, multi_class=multi_class, penalty=penalty, C=C)

            name = 'LR_solver_' + str(solver)

        elif model_name == 'RBM':

            learning_rate = hyperparameter['learning_rate']

            print("Performing RBM learning_rate: " + str(learning_rate) + " - K " + str(self.current_K))

            model = Binarizer()

            logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=10000, multi_class='multinomial')

            rbm = BernoulliRBM(random_state=0, verbose=True, n_components=500, n_iter=20, learning_rate=learning_rate)
            rbm2 = BernoulliRBM(random_state=0, verbose=True, n_components=500, n_iter=20, learning_rate=learning_rate)
            rbm3 = BernoulliRBM(random_state=0, verbose=True, n_components=500, n_iter=20, learning_rate=learning_rate)

            model = Pipeline(steps=[('rbm', rbm), ('rbm2', rbm2), ('rbm3', rbm3), ('logistic', logistic)])

            name = 'RBM_lr_' + str(learning_rate)

        elif model_name == 'Ridge':

            alpha = hyperparameter['alpha']
            Fit_intercept = hyperparameter['Fit_intercept']
            Normalize = hyperparameter['Normalize']
            print("Performing Ridge alpha: " + str(alpha) + " - K " + str(self.current_K))

            model = Ridge(random_state=0, alpha=alpha, fit_intercept=Fit_intercept, normalize=Normalize)

            name = 'Ridge_alpha_' + str(alpha)

        elif model_name == 'Lasso':

            alpha = hyperparameter['alpha']
            print("Performing Lasso alpha: " + str(alpha) + " - K " + str(self.current_K))

            model = Lasso(random_state=0, alpha=alpha)

            name = 'Lasso_alpha_' + str(alpha)

        elif model_name == 'DecisionTreeClassifier':

            max_depth = hyperparameter['max_depth']
            print("Performing DecisionTreeClassifier " + str(max_depth) + " - K " + str(self.current_K))
            model = tree.DecisionTreeClassifier(max_depth=max_depth)

            name = 'DecisionTreeClassifier' + str(max_depth)

        elif model_name == 'AdaBoostClassifier':

            n_estimators = hyperparameter['n_estimators']
            print("Performing AdaBoostClassifier " + str(n_estimators) + " - K " + str(self.current_K))
            model = ensemble.AdaBoostClassifier(n_estimators=n_estimators)

            name = 'AdaBoostClassifier' + str(n_estimators)

        elif model_name == 'GradientBoostingClassifier':

            n_estimators= hyperparameter['n_estimators']
            print("Performing GradientBoostingClassifier" + str(n_estimators) + " - K " + str(self.current_K))
            model = ensemble.GradientBoostingClassifier(n_estimators=n_estimators)

            name = 'GradientBoostingClassifier' + str(n_estimators)

        return model, name

    def perform_classification(self, model_names, use_accuracy=True):
        tic = time.time()

        df = pd.DataFrame(columns=['Valid accuracy', 'Test accuracy', 'Valid Pearson', 'Test Pearson', 'Test R2'])

        self.y_valid_K = {}
        self.y_valid_true = {}

        self.y_test_true = []
        self.y_test_pred = []

        model_name = model_names[0]

        load_data = Load_data(DB_N=self.DB_N, which_model='MLP', subject=self.subject)
        self.K_fold_I = load_data.K_fold_I

        self.K_fold = load_data.K_fold

        for model_name in model_names:

            hyperparameters = get_hyperparameters(model_name=model_name)

            accuracies = np.empty([len(hyperparameters), self.K_fold])
            pearson_coefs = np.empty([len(hyperparameters), self.K_fold])

            i = 0
            hyper = hyperparameters[i]
            K = 0

            for i, hyper in enumerate(hyperparameters):

                for K in range(self.K_fold):

                    self.current_K = K

                    X_train, X_valid, Y_train, Y_valid, Y_orig_train, Y_orig_valid = load_data.K_fold_data(
                        K=self.current_K)

                    s = Up_sampling(X=X_train, Y=Y_train)
                    X_train, Y_train = s.upsampling()

                    X = np.concatenate([X_train, X_valid], axis=0)
                    Y = np.concatenate([Y_train, Y_valid], axis=0)

                    train_ind = [0, X_train.shape[0]]

                    model, name = self.get_model(model_name=model_name, hyperparameter=hyper)

                    model.fit(X_train, Y_train)

                    Y_pred_oh_max = model.predict(X_valid)
                    self.y_valid_K[K] = Y_pred_oh_max
                    self.y_valid_true[K] = Y_valid

                    if model_name == 'RF_Regression' or 'GP'or 'Ridge':

                       Y_pred_oh_max = np.abs(np.round(Y_pred_oh_max)).squeeze()

                    accuracy = np.round(metrics.accuracy_score(Y_valid, Y_pred_oh_max), decimals=4)
                    #accuracy = np.round(compute_balanced_accuracy(Y_valid, Y_pred_oh_max), decimals=4)
                    corrcoef = np.corrcoef(Y_valid.squeeze(), Y_pred_oh_max.squeeze())[1, 0]

                    pearson_coefs[i, K] = corrcoef
                    accuracies[i, K] = accuracy

                accuracies_mean = np.mean(accuracies, axis=1)
                accuracies_std = np.std(accuracies, axis=1)

                pearson_coefs_mean = np.mean(pearson_coefs, axis=1)
                pearson_coefs_std = np.std(pearson_coefs, axis=1)

            def info_best(accuracies_mean, accuracies_std):

                best_index_acc = unravel_index(accuracies_mean.argmax(), accuracies_mean.shape)

                best_mean_acc = accuracies_mean[best_index_acc]
                best_std_acc = accuracies_std[best_index_acc]

                return best_index_acc, best_mean_acc, best_std_acc

            best_index_acc, best_mean_acc, best_std_acc = info_best(accuracies_mean, accuracies_std)
            best_index_pearson_coefs, best_mean_pearson_coefs, best_std_pearson_coefs = info_best(pearson_coefs_mean,
                                                                                                  pearson_coefs_std)

            if use_accuracy:

                mean_acc = best_mean_acc
                std_acc = best_std_acc
                best_index = best_index_acc

                mean_pearson_coefs = pearson_coefs_mean[best_index_acc[0]]
                std_pearson_coefs = pearson_coefs_std[best_index_acc[0]]

                print('Best Hyperparameter for acc:')
                print(hyperparameters[best_index_acc[0]])

            else:

                mean_pearson_coefs = best_mean_pearson_coefs
                std_pearson_coefs = best_std_pearson_coefs
                best_index = best_index_pearson_coefs

                mean_acc = accuracies_mean[best_index_pearson_coefs[0]]
                std_acc = accuracies_std[best_index_pearson_coefs[0]]

                print('Best Hyperparameter for pearson_coefs:')
                print(hyperparameters[best_index_pearson_coefs[0]])

            # Test
            X_train, X_valid, Y_train, Y_valid, Y_orig_train, Y_orig_valid = load_data.K_fold_data(K=0)
            X_test, Y_test, Y_orig_test = load_data.test_data()

            X = np.concatenate([X_train, X_test], axis=0)
            Y = np.concatenate([Y_train, Y_test], axis=0)

            train_ind = [0, X_train.shape[0]]

            if use_accuracy:
                hyper = hyperparameters[best_index_acc[0]]
                model, name = self.get_model(model_name=model_name, hyperparameter=hyper)
                print('Test by using best acc model')
                print('Best Hyperparameter for acc:')
                print(hyperparameters[best_index_acc[0]])

            else:

                hyper = hyperparameters[best_index_pearson_coefs[0]]
                model, name = self.get_model(model_name=model_name, hyperparameter=hyper)
                print(hyperparameters[best_index_pearson_coefs[0]])
                print('Test by using best pearson_coefs model')
                print('Best Hyperparameter for pearson_coefs:')
                print(hyperparameters[best_index_pearson_coefs[0]])

            model.fit(X_train, Y_train)

            y_test_pred = model.predict(X_test)

            self.y_test_true = Y_test
            self.y_test_pred = y_test_pred

            if model_name == 'RF_Regression' or 'GP':
                y_test_pred = np.abs(np.round(y_test_pred)).squeeze()

            accuracy = np.round(metrics.accuracy_score(Y_test, y_test_pred), decimals=4)
            #accuracy = np.round(compute_balanced_accuracy(Y_test, y_test_pred), decimals=4)
            corrcoef = np.corrcoef(Y_test.squeeze(), y_test_pred.squeeze())[1, 0]
            R2= r2_score(Y_test, y_test_pred)

            # Accuracy
            K_acc_str = str("{0:.2f}".format(mean_acc * 100)) + " (" + str("{0:.3}".format(std_acc * 100)) + ")"
            Test_acc_str = str("{0:.2f}".format(accuracy * 100))


            print("Valid Accuracy: " + K_acc_str)
            print("Test Accuracy: " + Test_acc_str)

            # Pearson
            K_pearson_str = str("{0:.2f}".format(mean_pearson_coefs)) + " (" + str("{0:.3}".format(std_pearson_coefs)) + ")"
            Test_pearson_str = str("{0:.2f}".format(corrcoef))

            print(' ')
            print("Valid Pearson: " + K_pearson_str)
            print("Test Pearson: " + Test_pearson_str)

            R2_str = str("{0:.2f}".format(R2))
            print("Test Pearson: " + R2_str)

            df.loc[name] = [K_acc_str, Test_acc_str, K_pearson_str, Test_pearson_str, R2_str]
            self.K_acc_str=K_acc_str
            self.Test_acc_str = Test_acc_str
            self.K_pearson_str = K_pearson_str
            self.Test_pearson_str = Test_pearson_str
            self.R2_str = R2_str

            toc = time.time()

            print(" - Trained in " + str(toc - tic) + " s")

            # name = "Y_valid_K_" + str(subject).zfill(2) + ".npy"
            # np.save(path_info.tensorboard_path_DB + "" + name, self.y_valid_K)
            # name = "Y_valid_K_true_" + str(subject).zfill(2) + ".npy"
            # np.save(path_info.tensorboard_path_DB + "" + name, self.y_valid_true)

        return df,Test_acc_str,Test_pearson_str

DB_N = 0
subject = 3

which_model = 'MLP'
path_info = Path_info(subject_ind=subject)
path_info.get_DB_path_2(DB_N=DB_N, delete_folder=False)
path_info.get_DB_info_2(DB_N=DB_N)

hypers_DA = [{'n': 0, 'random_noise_factor': 0.01, 'augmentation_types': 'rn'}]
hyper_DA = hypers_DA[0]  # it is for data augmentation

K_fold =5
create_DB = Create_DB(DB_N, hyper_DA)
#self = Create_DB(DB_N, hyper_DA)
accuracies = np.empty([K_fold])
pearson = np.empty([K_fold])
y_test_pred = {}
y_test_true = {}
#KK=0
import seaborn as sns
def confusion_matrix(cm, accuracy):
    index = ['bent_row', 'lat_raise', 'sh_press']
    columns = ['bent_row', 'lat_raise', 'sh_press']
    cm_df = pd.DataFrame(cm, columns, index)
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm_df, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
    plt.title(all_sample_title, size=15)
for KK in range(K_fold):
    create_DB.create_database(plotIt=False, KK=KK)

    model_names_classification = model_names_classification = ['SVM', 'LDA', 'QDA', 'RF', 'MLP', 'SVC', 'KNN_Classifier', 'LR', 'RBM','RF_Regression','Ridge','DecisionTreeClassifier','AdaBoostClassifier','GradientBoostingClassifier']
    model_names_classification = model_names_classification = ['RF']

    model_names = model_names_classification
    # self = Machine_Learning_classification(DB_N=DB_N, subject=subject)
    ml_class = Machine_Learning_classification(DB_N=DB_N, subject=subject)
    df_classification,Test_acc_str,Test_pearson_str = ml_class.perform_classification(model_names=model_names_classification, use_accuracy=True)
    y_test_pred[KK] = ml_class.y_test_pred
    y_test_true[KK] = ml_class.y_test_true
    print(df_classification)
    pearson[KK] = Test_pearson_str
    accuracies[KK] = Test_acc_str

accuracies_mean = np.mean(accuracies, axis=0)
accuracies_std = np.std(accuracies, axis=0)

pearson_coefs_mean = np.mean(pearson, axis=0)
pearson_coefs_std = np.std(pearson, axis=0)

# model_names_clustering = ['PCA', 'TSNE', 'ICA', 'ISO', 'laplacian', 'LDA']
# model_names_classification = ['SVM', 'LDA', 'QDA', 'RF', 'MLP', 'SVC', 'KNN_Classifier', 'LR', 'RBM','RF_Regression','Ridge','DecisionTreeClassifier','AdaBoostClassifier','GradientBoostingClassifier']

y_pred = np.concatenate([y_test_pred[K] for K in y_test_pred.keys()], axis=0)
y_true = np.concatenate([ y_test_true[K] for K in  y_test_true.keys()], axis=0)

# Confusion matrix
cm = metrics.confusion_matrix(y_true, y_pred)
print(cm)
cmxN = cm / cm.astype(np.float).sum(axis=0)
print(cmxN)
confusion_matrix(cm=cmxN, accuracy=metrics.accuracy_score(y_true, y_pred))

