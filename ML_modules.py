import time
import itertools
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
matplotlib.use('TKAgg')

from sklearn import decomposition, svm, linear_model, metrics, tree
from sklearn.metrics import r2_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE, MDS, LocallyLinearEmbedding, Isomap, SpectralEmbedding
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.preprocessing import Binarizer
from sklearn.decomposition import FastICA, PCA
from sklearn.neural_network import MLPClassifier, MLPRegressor, BernoulliRBM
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel,ConstantKernel, RBF

from Classes.Path import Path_info
from Classes.Data_2 import Load_data, Create_DB, Up_sampling


def info_best(accuracies_mean, accuracies_std):
    best_index_acc = np.unravel_index(accuracies_mean.argmax(), accuracies_mean.shape)

    best_mean_acc = accuracies_mean[best_index_acc]
    best_std_acc = accuracies_std[best_index_acc]

    return best_index_acc, best_mean_acc, best_std_acc


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

    # elif model_name == 'MLP':
    #     tuned_parameters = {'hidden_layer_sizes': [(64), (64, 128), (64, 128, 256), (32), (32, 64), (32, 64, 128)],
    #                         'activation': ['tanh', 'relu', 'logistic'], 'solver': ['adam', 'lbfgs', 'sgd']}
    elif model_name == 'MLP':
        tuned_parameters = {'hidden_layer_sizes': [(64),(64,128)],
                            'activation': ['relu'], 'solver': ['adam']}

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
    for vals in itertools.product(*tuned_parameters.values()):
        hyperparameters.append(dict(zip(tuned_parameters, vals)))

    return hyperparameters


class Machine_Learning_classification():

    def __init__(self, DB_N):

        self.results = {}
        self.results_train = {}
        self.results_test = {}

        self.results_classification = {}
        self.DB_N = DB_N

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

            model = svm.SVC(kernel=kernel)

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
            model = AdaBoostClassifier(n_estimators=n_estimators)

            name = 'AdaBoostClassifier' + str(n_estimators)

        elif model_name == 'GradientBoostingClassifier':

            n_estimators= hyperparameter['n_estimators']
            print("Performing GradientBoostingClassifier" + str(n_estimators) + " - K " + str(self.current_K))
            model = GradientBoostingClassifier(n_estimators=n_estimators)

            name = 'GradientBoostingClassifier' + str(n_estimators)

        return model, name

    def perform_classification(self, model_names, hyper_DA, K_folds=3, use_accuracy=True):
        tic = time.time()

        df = pd.DataFrame(columns=['Valid accuracy', 'Test accuracy', 'Valid Pearson', 'Test Pearson', 'Test R2'])

        self.y_valid_K = {}
        self.y_valid_true = {}

        self.y_test_true = []
        self.y_test_pred = []

        create_DB = Create_DB(self.DB_N, hyper_DA)
        create_DB.create_database(plotIt=False)
        load_data = Load_data(DB_N=self.DB_N, which_model='MLP', K_tot=K_folds)

        for model_name in model_names:
            print('MODEL:',model_name)

            hyperparameters = get_hyperparameters(model_name=model_name)

            accuracies = np.empty([len(hyperparameters), K_folds]) # 5 is the inner loop
            pearson_coefs = np.empty([len(hyperparameters), K_folds])
            test_acc=np.empty([K_folds])
            test_pear = np.empty([K_folds])

            for i, hyper in enumerate(hyperparameters):
                print('\nVERSION:',i)

                y_test_predict = {}
                y_test_true = {}

                for K in range(K_folds):

                    self.current_K = K

                    X_train, X_valid, Y_train, Y_valid = load_data.K_fold_data(
                        K=self.current_K)

                    s = Up_sampling(X=X_train, Y=Y_train)
                    X_train, Y_train = s.upsampling()

                    X = np.concatenate([X_train, X_valid], axis=0)
                    Y = np.concatenate([Y_train, Y_valid], axis=0)

                    train_ind = [0, X_train.shape[0]]

                    model, name = self.get_model(model_name=model_name, hyperparameter=hyper)

                    model.fit(X_train, Y_train)

                    Y_pred_oh_max = model.predict(X_valid)

                    if model_name == 'RF_Regression' or 'GP' or 'Ridge':
                        Y_pred_oh_max = np.abs(np.round(Y_pred_oh_max)).squeeze()

                    accuracy = np.round(metrics.accuracy_score(Y_valid, Y_pred_oh_max), decimals=4)
                    corrcoef = np.corrcoef(Y_valid.squeeze(), Y_pred_oh_max.squeeze())[1, 0]

                    pearson_coefs[i, K] = corrcoef
                    accuracies[i, K] = accuracy

                    ################ TESTING ##############
                    X_test, Y_test = load_data.test_data()
                    y_test_pred = model.predict(X_test)
                    if model_name == 'RF_Regression' or 'GP':
                        y_test_pred = np.abs(np.round(y_test_pred)).squeeze()

                    y_test_predict[K] = y_test_pred
                    y_test_true[K] = Y_test

                    accuracy = np.round(metrics.accuracy_score(Y_test, y_test_pred), decimals=4)
                    # accuracy = np.round(compute_balanced_accuracy(Y_test, y_test_pred), decimals=4)
                    corrcoef = np.corrcoef(Y_test.squeeze(), y_test_pred.squeeze())[1, 0]

                    test_pear[K] = corrcoef
                    test_acc[K] = accuracy

                accuracies_mean = np.mean(accuracies, axis=1)
                accuracies_std = np.std(accuracies, axis=1)

                pearson_coefs_mean = np.mean(pearson_coefs, axis=1)
                pearson_coefs_std = np.std(pearson_coefs, axis=1)

                best_index_acc, best_mean_acc, best_std_acc = info_best(accuracies_mean, accuracies_std)
                best_index_pearson_coefs, best_mean_pearson_coefs, best_std_pearson_coefs = info_best(
                    pearson_coefs_mean,
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

                y_pred = np.concatenate([y_test_predict[K] for K in y_test_predict.keys()], axis=0)
                y_true = np.concatenate([y_test_true[K] for K in y_test_true.keys()], axis=0)

                acc_mean = np.mean(test_acc, axis=0)
                acc_std = np.std(test_acc, axis=0)

                pear_mean = np.mean(test_pear, axis=0)
                pear_std = np.std(test_pear, axis=0)

                R2 = r2_score(y_pred, y_true)

                # validation Accuracy
                K_acc_str = str("{0:.2f}".format(mean_acc * 100)) + " (" + str(
                    "{0:.3}".format(std_acc * 100)) + ")"
                Test_acc_str =str("{0:.2f}".format(acc_mean * 100)) + " (" + str(
                    "{0:.3}".format(acc_std * 100)) + ")"

                print("Valid Accuracy: " + K_acc_str)
                print("Test Accuracy: " + Test_acc_str)

                # val Pearson
                K_pearson_str = str("{0:.2f}".format(mean_pearson_coefs)) + " (" + str(
                    "{0:.3}".format(std_pearson_coefs)) + ")"
                Test_pearson_str = str("{0:.2f}".format(pear_mean)) + " (" + str(
                    "{0:.3}".format(pear_std)) + ")"

                print(' ')
                print("Valid Pearson: " + K_pearson_str)
                print("Test Pearson: " + Test_pearson_str)

                R2_str = str("{0:.2f}".format(R2))
                print("R2: " + R2_str)

                df.loc[name] = [K_acc_str, Test_acc_str, K_pearson_str, Test_pearson_str, R2_str]
                # df.loc[name] = [K_acc_str,  K_pearson_str]
                self.K_acc_str = K_acc_str
                self.Test_acc_str = Test_acc_str
                self.K_pearson_str = K_pearson_str
                self.Test_pearson_str = Test_pearson_str
                self.R2_str = R2_str

                toc = time.time()

                print(" - Trained in " + str(toc - tic) + " s")

            print('#'*100)

        print('#'*100)

        return df, Test_acc_str, Test_pearson_str, y_pred, y_true