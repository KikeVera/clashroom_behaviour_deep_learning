import pickle

from matplotlib import pyplot as plt

from ml_utils.machine_learning_utils import MachineLearningUtils


def run_wisdm_knn_train(wisdm_train_df, path, hp=None, external_test=None):
    ml_utils = MachineLearningUtils()
    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=wisdm_train_df,
                                                                  predicted_var='action')
    if external_test is not None:
        x_test, y_test = ml_utils.get_external_test(train_df=external_test,
                                                    predicted_var='action')

    knn = ml_utils.knn_train(x_train, x_test, y_train, y_test, hp=hp)

    pickle.dump(knn, open(path + "knn_model.pkl", "wb"))

    ml_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=knn, model_name="KNN")


def run_wisdm_mlp_train(wisdm_train_df, path, hp=None, external_test=None):
    ml_utils = MachineLearningUtils()
    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=wisdm_train_df,
                                                                  predicted_var='action')
    if external_test is not None:
        x_test, y_test = ml_utils.get_external_test(train_df=external_test,
                                                    predicted_var='action')
    mlp = ml_utils.neuronal_network_train(x_train, x_test, y_train, y_test, hp=hp)

    pickle.dump(mlp, open(path + "neural_network_model.pkl", "wb"))

    ml_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=mlp,
                                   model_name="Neural Network")


def run_wisdm_rf_train(wisdm_train_df, path, hp=None, external_test=None):
    ml_utils = MachineLearningUtils()
    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=wisdm_train_df,
                                                                  predicted_var='action')
    if external_test is not None:
        x_test, y_test = ml_utils.get_external_test(train_df=external_test,
                                                    predicted_var='action')
    rf = ml_utils.random_forest_train(x_train, x_test, y_train, y_test, hp=hp)

    pickle.dump(rf, open(path + "random_forest_model.pkl", "wb"))

    ml_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=rf,
                                   model_name="Random Forest")


def run_wisdm_xtree_train(wisdm_train_df, path, hp=None, external_test=None):
    ml_utils = MachineLearningUtils()
    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=wisdm_train_df,
                                                                  predicted_var='action')
    if external_test is not None:
        x_test, y_test = ml_utils.get_external_test(train_df=external_test,
                                                    predicted_var='action')

    xtree = ml_utils.extra_tree_train(x_train, x_test, y_train, y_test, hp=hp)

    pickle.dump(xtree, open(path + "xtree_model.pkl", "wb"))

    ml_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=xtree, model_name="Xtree")


def run_wisdm_decision_tree_train(wisdm_train_df, path, hp=None, external_test=None):
    ml_utils = MachineLearningUtils()
    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=wisdm_train_df,
                                                                  predicted_var='action')
    if external_test is not None:
        x_test, y_test = ml_utils.get_external_test(train_df=external_test,
                                                    predicted_var='action')

    decision_tree = ml_utils.decision_tree_train(x_train, x_test, y_train, y_test, hp=hp)

    pickle.dump(decision_tree, open(path + "decision_tree_model.pkl", "wb"))

    ml_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=decision_tree, model_name="Xtree")


def run_wisdm_svm_train(wisdm_train_df, path, hp=None, external_test=None):
    ml_utils = MachineLearningUtils()
    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=wisdm_train_df,
                                                                  predicted_var='action')
    if external_test is not None:
        x_test, y_test = ml_utils.get_external_test(train_df=external_test,
                                                    predicted_var='action')

    svm = ml_utils.svm_train(x_train, x_test, y_train, y_test, hp=hp)

    pickle.dump(svm, open(path + "svm_model.pkl", "wb"))

    ml_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=svm, model_name="Xtree")


def run_wisdm_bayes_train(wisdm_train_df, path, hp=None, external_test=None):
    ml_utils = MachineLearningUtils()
    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=wisdm_train_df,
                                                                  predicted_var='action')
    if external_test is not None:
        x_test, y_test = ml_utils.get_external_test(train_df=external_test,
                                                    predicted_var='action')

    bayes = ml_utils.bayes_train(x_train, x_test, y_train, y_test, hp=hp)

    pickle.dump(bayes, open(path + "bayes_model.pkl", "wb"))

    ml_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=bayes, model_name="Xtree")


def run_wisdm_stochastic_gradient_train(wisdm_train_df, path, hp=None, external_test=None):
    ml_utils = MachineLearningUtils()
    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=wisdm_train_df,
                                                                  predicted_var='action')
    if external_test is not None:
        x_test, y_test = ml_utils.get_external_test(train_df=external_test,
                                                    predicted_var='action')

    stochastic_gradient = ml_utils.stochastic_gradient_train(x_train, x_test, y_train, y_test, hp=hp)

    pickle.dump(stochastic_gradient, open(path + "stochastic_gradient_model.pkl", "wb"))

    ml_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=stochastic_gradient, model_name="Xtree")


def run_wisdm_train_stacking(wisdm_train_df, path, external_test=None, stack_features=None, mlp_knn_rf_features=None,
                             mlp_knn_xtree_features=None,
                             xtree_knn_rf_features=None, mlp_xtree_rf_features=None, mlp_knn_features=None,
                             rf_knn_features=None,
                             rf_mlp_features=None, xtree_mlp_features=None, xtree_knn_features=None,
                             xtree_rf_features=None):
    ml_utils = MachineLearningUtils()

    train_df = wisdm_train_df[stack_features] if stack_features is not None else wisdm_train_df
    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=train_df,
                                                                  predicted_var='action')
    if external_test is not None:
        test_df = external_test[stack_features] if stack_features is not None else external_test
        x_test, y_test = ml_utils.get_external_test(train_df=test_df,
                                                    predicted_var='action')
    stack = ml_utils.stacking_train(x_train, x_test, y_train, y_test)
    pickle.dump(stack, open(path + "stacking_model.pkl", "wb"))

    train_df = wisdm_train_df[mlp_knn_rf_features] if mlp_knn_rf_features is not None else wisdm_train_df
    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=train_df,
                                                                  predicted_var='action')
    if external_test is not None:
        test_df = external_test[mlp_knn_rf_features] if mlp_knn_rf_features is not None else external_test
        x_test, y_test = ml_utils.get_external_test(train_df=test_df,
                                                    predicted_var='action')


    mlp_knn_rf = ml_utils.stacking_train(x_train, x_test, y_train, y_test, xtree=False)
    pickle.dump(mlp_knn_rf, open(path + "mlp_knn_rf_model.pkl", "wb"))

    train_df = wisdm_train_df[mlp_knn_xtree_features] if mlp_knn_xtree_features is not None else wisdm_train_df
    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=train_df,
                                                                  predicted_var='action')
    if external_test is not None:
        test_df = external_test[mlp_knn_xtree_features] if mlp_knn_xtree_features is not None else external_test
        x_test, y_test = ml_utils.get_external_test(train_df=test_df,
                                                    predicted_var='action')

    mlp_knn_xtree = ml_utils.stacking_train(x_train, x_test, y_train, y_test, rf=False)
    pickle.dump(mlp_knn_xtree, open(path + "mlp_knn_xtree_model.pkl", "wb"))

    train_df = wisdm_train_df[xtree_knn_rf_features] if xtree_knn_rf_features is not None else wisdm_train_df

    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=train_df,
                                                                  predicted_var='action')
    if external_test is not None:
        test_df = external_test[xtree_knn_rf_features] if xtree_knn_rf_features is not None else external_test
        x_test, y_test = ml_utils.get_external_test(train_df=test_df,
                                                    predicted_var='action')

    xtree_knn_rf = ml_utils.stacking_train(x_train, x_test, y_train, y_test, mlp=False)
    pickle.dump(xtree_knn_rf, open(path + "xtree_knn_rf_model.pkl", "wb"))

    train_df = wisdm_train_df[mlp_xtree_rf_features] if mlp_xtree_rf_features is not None else wisdm_train_df

    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=train_df,
                                                                  predicted_var='action')
    if external_test is not None:
        test_df = external_test[mlp_xtree_rf_features] if mlp_xtree_rf_features is not None else external_test
        x_test, y_test = ml_utils.get_external_test(train_df=test_df,
                                                    predicted_var='action')

    mlp_xtree_rf = ml_utils.stacking_train(x_train, x_test, y_train, y_test, knn=False)
    pickle.dump(mlp_xtree_rf, open(path + "mlp_xtree_rf_model.pkl", "wb"))

    train_df = wisdm_train_df[mlp_knn_features] if mlp_knn_features is not None else wisdm_train_df

    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=train_df,
                                                                  predicted_var='action')
    if external_test is not None:
        test_df = external_test[mlp_knn_features] if mlp_knn_features is not None else external_test
        x_test, y_test = ml_utils.get_external_test(train_df=test_df,
                                                    predicted_var='action')

    mlp_knn = ml_utils.stacking_train(x_train, x_test, y_train, y_test, rf=False, xtree=False)
    pickle.dump(mlp_knn, open(path + "mlp_knn_model.pkl", "wb"))

    train_df = wisdm_train_df[rf_knn_features] if rf_knn_features is not None else wisdm_train_df

    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=train_df,
                                                                  predicted_var='action')
    if external_test is not None:
        test_df = external_test[rf_knn_features] if rf_knn_features is not None else external_test
        x_test, y_test = ml_utils.get_external_test(train_df=test_df,
                                                    predicted_var='action')

    rf_knn = ml_utils.stacking_train(x_train, x_test, y_train, y_test, mlp=False, xtree=False)
    pickle.dump(rf_knn, open(path + "rf_knn_model.pkl", "wb"))

    train_df = wisdm_train_df[rf_mlp_features] if rf_mlp_features is not None else wisdm_train_df
    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=train_df,
                                                                  predicted_var='action')
    if external_test is not None:
        test_df = external_test[rf_mlp_features] if rf_mlp_features is not None else external_test
        x_test, y_test = ml_utils.get_external_test(train_df=test_df,
                                                    predicted_var='action')

    rf_mlp = ml_utils.stacking_train(x_train, x_test, y_train, y_test, knn=False, xtree=False)
    pickle.dump(rf_mlp, open(path + "rf_mlp_model.pkl", "wb"))

    train_df = wisdm_train_df[xtree_mlp_features] if xtree_mlp_features is not None else wisdm_train_df
    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=train_df,
                                                                  predicted_var='action')
    if external_test is not None:
        test_df = external_test[xtree_mlp_features] if xtree_mlp_features is not None else external_test
        x_test, y_test = ml_utils.get_external_test(train_df=test_df,
                                                    predicted_var='action')

    xtree_mlp = ml_utils.stacking_train(x_train, x_test, y_train, y_test, knn=False, rf=False)
    pickle.dump(xtree_mlp, open(path + "xtree_mlp_model.pkl", "wb"))

    train_df = wisdm_train_df[xtree_knn_features] if xtree_knn_features is not None else wisdm_train_df
    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=train_df,
                                                                  predicted_var='action')
    if external_test is not None:
        test_df = external_test[xtree_knn_features] if xtree_knn_features is not None else external_test
        x_test, y_test = ml_utils.get_external_test(train_df=test_df,
                                                    predicted_var='action')

    xtree_knn = ml_utils.stacking_train(x_train, x_test, y_train, y_test, rf=False, mlp=False)
    pickle.dump(xtree_knn, open(path + "xtree_knn_model.pkl", "wb"))

    train_df = wisdm_train_df[xtree_rf_features] if xtree_rf_features is not None else wisdm_train_df
    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=train_df,
                                                                  predicted_var='action')
    if external_test is not None:
        test_df = external_test[xtree_rf_features] if xtree_rf_features is not None else external_test
        x_test, y_test = ml_utils.get_external_test(train_df=test_df,
                                                    predicted_var='action')

    xtree_rf = ml_utils.stacking_train(x_train, x_test, y_train, y_test, mlp=False, knn=False)
    pickle.dump(xtree_rf, open(path + "xtree_rf_model.pkl", "wb"))
    #
    # ml_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=stack,
    #                                                   model_name="mlp_knn_rf_xtree")
    # ml_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=mlp_knn_rf,
    #                                                   model_name="mlp_knn_rf")
    # ml_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=mlp_xtree_rf,
    #                                                   model_name="mlp_xtree_rf")
    # ml_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=mlp_knn_xtree,
    #                                                   model_name="mlp_knn_xtree")
    # ml_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=xtree_knn_rf,
    #                                                   model_name="xtree_knn_rf")
    # ml_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=mlp_knn,
    #                                                   model_name="mlp_knn")
    # ml_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=rf_mlp,
    #                                                   model_name="rf_mlp")
    # ml_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=rf_knn,
    #                                                   model_name="rf_knn")
    # ml_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=xtree_rf,
    #                                                   model_name="xtree_rf")
    # ml_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=xtree_mlp,
    #                                                   model_name="xtree_mlp")
    # ml_utils.plot_confusion_matrix(x_test=x_test, y_test=y_test, model=xtree_knn,
    #                                                   model_name="xtree_knn")

    # plt.show()
