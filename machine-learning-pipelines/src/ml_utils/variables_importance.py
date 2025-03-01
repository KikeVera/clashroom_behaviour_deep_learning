import os

from sklearn.inspection import permutation_importance

from ml_utils.machine_learning_utils import MachineLearningUtils
from ml_utils.train_wisdm_dataframe import run_wisdm_knn_train, run_wisdm_mlp_train, run_wisdm_rf_train, \
    run_wisdm_xtree_train, run_wisdm_train_stacking

knn_variables = ['action', 'mean_ax', 'std_ax', 'max_ax', 'min_ax', 'var_ax', 'median_ax', 'mean_peak_values_ax',
                 'std_peak_values_ax', 'max_peak_values_ax', 'min_peak_values_ax', 'var_peak_values_ax', 'p25_ax',
                 'p50_ax', 'p75_ax', 'mean_ay', 'std_ay', 'max_ay', 'min_ay', 'var_ay', 'median_ay',
                 'mean_peak_values_ay', 'std_peak_values_ay', 'max_peak_values_ay', 'min_peak_values_ay',
                 'var_peak_values_ay', 'p25_ay', 'p50_ay', 'p75_ay', 'std_az', 'max_az', 'min_az', 'var_az',
                 'median_az', 'std_peak_values_az', 'max_peak_values_az', 'min_peak_values_az', 'var_peak_values_az',
                 'p25_az', 'p50_az', 'mean_gx', 'std_gx', 'min_gx', 'var_gx', 'mean_peak_values_gx',
                 'std_peak_values_gx', 'min_peak_values_gx', 'var_peak_values_gx', 'p25_gx', 'p50_gx', 'mean_gy',
                 'std_gy', 'max_gy', 'min_gy', 'mean_peak_values_gy', 'std_peak_values_gy', 'max_peak_values_gy',
                 'min_peak_values_gy', 'var_peak_values_gy', 'p75_gy', 'std_gz', 'max_gz', 'min_gz', 'var_gz',
                 'median_gz', 'std_peak_values_gz', 'max_peak_values_gz', 'min_peak_values_gz', 'var_peak_values_gz',
                 'p25_gz', 'p50_gz']

mlp_variables = ['action', 'mean_ax', 'std_ax', 'max_ax', 'min_ax', 'var_ax', 'median_ax', 'mean_peak_values_ax',
                 'std_peak_values_ax', 'max_peak_values_ax', 'min_peak_values_ax', 'var_peak_values_ax', 'p25_ax',
                 'p50_ax', 'p75_ax', 'mean_ay', 'std_ay', 'max_ay', 'min_ay', 'var_ay', 'median_ay',
                 'mean_peak_values_ay', 'std_peak_values_ay', 'max_peak_values_ay', 'min_peak_values_ay',
                 'var_peak_values_ay', 'p25_ay', 'p50_ay', 'mean_az', 'std_az', 'max_az', 'min_az', 'var_az',
                 'mean_peak_values_az', 'std_peak_values_az', 'max_peak_values_az', 'min_peak_values_az', 'p25_az',
                 'p50_az', 'mean_gx', 'std_gx', 'max_gx', 'min_gx', 'mean_peak_values_gx', 'std_peak_values_gx',
                 'max_peak_values_gx', 'min_peak_values_gx', 'var_peak_values_gx', 'p50_gx', 'mean_gy', 'std_gy',
                 'max_gy', 'min_gy', 'var_gy', 'mean_peak_values_gy', 'std_peak_values_gy', 'max_peak_values_gy',
                 'min_peak_values_gy', 'var_peak_values_gy', 'p25_gy', 'p50_gy', 'mean_gz', 'std_gz', 'max_gz',
                 'min_gz', 'var_gz', 'mean_peak_values_gz', 'std_peak_values_gz', 'max_peak_values_gz',
                 'min_peak_values_gz', 'p25_gz', 'p50_gz']

rf_variables = ['action', 'max_ax', 'max_peak_values_ax', 'var_peak_values_ax', 'mean_ay', 'var_ay', 'median_ay',
                'max_peak_values_ay', 'p25_ay', 'p50_ay', 'p75_ay', 'var_peak_values_az', 'p25_az', 'std_gx',
                'std_peak_values_gx', 'var_peak_values_gx', 'p50_gx', 'p75_gx', 'var_peak_values_gy', 'p50_gy',
                'p75_gy', 'var_gz', 'var_peak_values_gz', 'p25_gz', 'p50_gz']

xtree_variables = ['action', 'std_ax', 'max_ax', 'var_ax', 'median_ax', 'mean_peak_values_ax', 'std_peak_values_ax',
                   'max_peak_values_ax', 'min_peak_values_ax', 'var_peak_values_ax', 'p25_ax', 'p50_ax', 'mean_ay',
                   'std_ay', 'max_ay', 'min_ay', 'median_ay', 'mean_peak_values_ay', 'max_peak_values_ay',
                   'var_peak_values_ay', 'p25_ay', 'p50_ay', 'p75_ay', 'mean_az', 'std_az', 'var_az', 'median_az',
                   'max_peak_values_az', 'var_peak_values_az', 'p25_az', 'p50_az', 'std_gx', 'mean_peak_values_gx',
                   'std_peak_values_gx', 'min_peak_values_gx', 'var_peak_values_gx', 'p25_gx', 'p50_gx', 'mean_gy',
                   'max_gy', 'min_gy', 'median_gy', 'max_peak_values_gy', 'min_peak_values_gy', 'var_peak_values_gy',
                   'p50_gy', 'p75_gy', 'mean_gz', 'std_gz', 'var_gz', 'median_gz', 'mean_peak_values_gz',
                   'max_peak_values_gz', 'p25_gz', 'p50_gz']


def variables_importance(train_df, test_df, sub_path):
    ml_utils = MachineLearningUtils()
    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=train_df, predicted_var='action')
    if test_df is not None:
        x_test, y_test = ml_utils.get_external_test(train_df=test_df, predicted_var='action')
    knn = ml_utils.knn_train(x_train, x_test, y_train, y_test)
    mlp = ml_utils.neuronal_network_train(x_train, x_test, y_train, y_test)
    rf = ml_utils.random_forest_train(x_train, x_test, y_train, y_test)
    xtree = ml_utils.extra_tree_train(x_train, x_test, y_train, y_test)

    stack = ml_utils.stacking_train(x_train, x_test, y_train, y_test)
    mlp_knn_rf = ml_utils.stacking_train(x_train, x_test, y_train, y_test, xtree=False)
    mlp_knn_xtree = ml_utils.stacking_train(x_train, x_test, y_train, y_test, rf=False)
    xtree_knn_rf = ml_utils.stacking_train(x_train, x_test, y_train, y_test, mlp=False)
    mlp_xtree_rf = ml_utils.stacking_train(x_train, x_test, y_train, y_test, knn=False)
    mlp_knn = ml_utils.stacking_train(x_train, x_test, y_train, y_test, rf=False, xtree=False)
    rf_knn = ml_utils.stacking_train(x_train, x_test, y_train, y_test, mlp=False, xtree=False)
    rf_mlp = ml_utils.stacking_train(x_train, x_test, y_train, y_test, knn=False, xtree=False)
    xtree_mlp = ml_utils.stacking_train(x_train, x_test, y_train, y_test, knn=False, rf=False)
    xtree_knn = ml_utils.stacking_train(x_train, x_test, y_train, y_test, rf=False, mlp=False)
    xtree_rf = ml_utils.stacking_train(x_train, x_test, y_train, y_test, mlp=False, knn=False)

    for importance_cut in [0]:
        importance_cut = importance_cut

        result = permutation_importance(knn, x_test, y_test, n_repeats=10, random_state=42)
        # Obtener importancia de características
        importance = result.importances_mean
        # Asociar importancia con nombres de características
        feature_names = train_df.columns  # Reemplaza con tus nombres de características
        feature_importance_dict = dict(zip(feature_names, importance))
        # Imprimir importancia de características

        knn_features = [feature for feature, importance in feature_importance_dict.items() if
                        importance > importance_cut]

        result = permutation_importance(mlp, x_test, y_test, n_repeats=10, random_state=42)
        # Obtener importancia de características
        importance = result.importances_mean
        # Asociar importancia con nombres de características
        feature_names = train_df.columns  # Reemplaza con tus nombres de características
        feature_importance_dict = dict(zip(feature_names, importance))
        # Imprimir importancia de características

        mlp_features = [feature for feature, importance in feature_importance_dict.items() if
                        importance > importance_cut]

        result = permutation_importance(rf, x_test, y_test, n_repeats=10, random_state=42)
        # Obtener importancia de características
        importance = result.importances_mean
        # Asociar importancia con nombres de características
        feature_names = train_df.columns  # Reemplaza con tus nombres de características
        feature_importance_dict = dict(zip(feature_names, importance))
        # Imprimir importancia de características

        rf_features = [feature for feature, importance in feature_importance_dict.items() if
                       importance > importance_cut]

        result = permutation_importance(xtree, x_test, y_test, n_repeats=10, random_state=42)
        # Obtener importancia de características
        importance = result.importances_mean
        # Asociar importancia con nombres de características
        feature_names = train_df.columns  # Reemplaza con tus nombres de características
        feature_importance_dict = dict(zip(feature_names, importance))
        # Imprimir importancia de características

        xtree_features = [feature for feature, importance in feature_importance_dict.items() if
                          importance > importance_cut]

        result = permutation_importance(stack, x_test, y_test, n_repeats=10, random_state=42)
        # Obtener importancia de características
        importance = result.importances_mean
        # Asociar importancia con nombres de características
        feature_names = train_df.columns  # Reemplaza con tus nombres de características
        feature_importance_dict = dict(zip(feature_names, importance))
        # Imprimir importancia de características

        stack_features = [feature for feature, importance in feature_importance_dict.items()
                          if importance > importance_cut]

        result = permutation_importance(mlp_knn_rf, x_test, y_test, n_repeats=10, random_state=42)
        # Obtener importancia de características
        importance = result.importances_mean
        # Asociar importancia con nombres de características
        feature_names = train_df.columns  # Reemplaza con tus nombres de características
        feature_importance_dict = dict(zip(feature_names, importance))
        # Imprimir importancia de características

        mlp_knn_rf_features = [feature for feature, importance in feature_importance_dict.items() if
                               importance > importance_cut]

        result = permutation_importance(mlp_knn_xtree, x_test, y_test, n_repeats=10, random_state=42)
        # Obtener importancia de características
        importance = result.importances_mean
        # Asociar importancia con nombres de características
        feature_names = train_df.columns  # Reemplaza con tus nombres de características
        feature_importance_dict = dict(zip(feature_names, importance))
        # Imprimir importancia de características

        mlp_knn_xtree_features = [feature for feature, importance in feature_importance_dict.items() if
                                  importance > importance_cut]

        result = permutation_importance(xtree_knn_rf, x_test, y_test, n_repeats=10, random_state=42)
        # Obtener importancia de características
        importance = result.importances_mean
        # Asociar importancia con nombres de características
        feature_names = train_df.columns  # Reemplaza con tus nombres de características
        feature_importance_dict = dict(zip(feature_names, importance))
        # Imprimir importancia de características

        xtree_knn_rf_features = [feature for feature, importance in feature_importance_dict.items() if
                                 importance > importance_cut]

        result = permutation_importance(mlp_xtree_rf, x_test, y_test, n_repeats=10, random_state=42)
        # Obtener importancia de características
        importance = result.importances_mean
        # Asociar importancia con nombres de características
        feature_names = train_df.columns  # Reemplaza con tus nombres de características
        feature_importance_dict = dict(zip(feature_names, importance))
        # Imprimir importancia de características

        mlp_xtree_rf_features = [feature for feature, importance in feature_importance_dict.items() if
                                 importance > importance_cut]

        result = permutation_importance(mlp_knn, x_test, y_test, n_repeats=10, random_state=42)
        # Obtener importancia de características
        importance = result.importances_mean
        # Asociar importancia con nombres de características
        feature_names = train_df.columns  # Reemplaza con tus nombres de características
        feature_importance_dict = dict(zip(feature_names, importance))
        # Imprimir importancia de características

        mlp_knn_features = [feature for feature, importance in feature_importance_dict.items() if
                            importance > importance_cut]

        result = permutation_importance(rf_knn, x_test, y_test, n_repeats=10, random_state=42)
        # Obtener importancia de características
        importance = result.importances_mean
        # Asociar importancia con nombres de características
        feature_names = train_df.columns  # Reemplaza con tus nombres de características
        feature_importance_dict = dict(zip(feature_names, importance))
        # Imprimir importancia de características

        rf_knn_features = [feature for feature, importance in feature_importance_dict.items() if
                           importance > importance_cut]

        result = permutation_importance(rf_mlp, x_test, y_test, n_repeats=10, random_state=42)
        # Obtener importancia de características
        importance = result.importances_mean
        # Asociar importancia con nombres de características
        feature_names = train_df.columns  # Reemplaza con tus nombres de características
        feature_importance_dict = dict(zip(feature_names, importance))
        # Imprimir importancia de características

        rf_mlp_features = [feature for feature, importance in feature_importance_dict.items() if
                           importance > importance_cut]

        result = permutation_importance(xtree_mlp, x_test, y_test, n_repeats=10, random_state=42)
        # Obtener importancia de características
        importance = result.importances_mean
        # Asociar importancia con nombres de características
        feature_names = train_df.columns  # Reemplaza con tus nombres de características
        feature_importance_dict = dict(zip(feature_names, importance))
        # Imprimir importancia de características

        xtree_mlp_features = [feature for feature, importance in feature_importance_dict.items() if
                              importance > importance_cut]

        result = permutation_importance(xtree_knn, x_test, y_test, n_repeats=10, random_state=42)
        # Obtener importancia de características
        importance = result.importances_mean
        # Asociar importancia con nombres de características
        feature_names = train_df.columns  # Reemplaza con tus nombres de características
        feature_importance_dict = dict(zip(feature_names, importance))
        # Imprimir importancia de características

        xtree_knn_features = [feature for feature, importance in feature_importance_dict.items() if
                              importance > importance_cut]

        result = permutation_importance(xtree_rf, x_test, y_test, n_repeats=10, random_state=42)
        # Obtener importancia de características
        importance = result.importances_mean
        # Asociar importancia con nombres de características
        feature_names = train_df.columns  # Reemplaza con tus nombres de características
        feature_importance_dict = dict(zip(feature_names, importance))
        # Imprimir importancia de características

        xtree_rf_features = [feature for feature, importance in feature_importance_dict.items() if
                             importance > importance_cut]

        variables = (knn_features, mlp_features, rf_features, xtree_features, stack_features, mlp_knn_rf_features,
                     mlp_knn_xtree_features, xtree_knn_rf_features, mlp_xtree_rf_features, mlp_knn_features,
                     rf_knn_features,
                     rf_mlp_features, xtree_mlp_features, xtree_knn_features, xtree_rf_features)

        for variable in variables:
            if "action" not in variable:
                variable.append("action")

        (knn_features, mlp_features, rf_features, xtree_features, stack_features, mlp_knn_rf_features,
         mlp_knn_xtree_features, xtree_knn_rf_features, mlp_xtree_rf_features, mlp_knn_features,
         rf_knn_features,
         rf_mlp_features, xtree_mlp_features, xtree_knn_features, xtree_rf_features) = variables

        path = sub_path + "data/models/wisdm_single/relevant_variables/" + str(importance_cut) + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        print("RUN WISDM SINGLE MODELS RELEVANT VARIABLES:" + str(importance_cut))
        test = test_df[knn_features] if test_df is not None else None
        run_wisdm_knn_train(wisdm_train_df=train_df[knn_features], path=path, external_test=test)
        test = test_df[mlp_features] if test_df is not None else None
        run_wisdm_mlp_train(wisdm_train_df=train_df[mlp_features], path=path, external_test=test)
        test = test_df[rf_features] if test_df is not None else None
        run_wisdm_rf_train(wisdm_train_df=train_df[rf_features], path=path, external_test=test)
        test = test_df[xtree_features] if test_df is not None else None
        run_wisdm_xtree_train(wisdm_train_df=train_df[xtree_features], path=path, external_test=test)

        path = sub_path + "data/models/wisdm_stack/relevant_variables/" + str(importance_cut) + "/"
        if not os.path.exists(path):
            os.makedirs(path)

        run_wisdm_train_stacking(train_df,
                                 path, test_df, stack_features, mlp_knn_rf_features,
                                 mlp_knn_xtree_features, xtree_knn_rf_features, mlp_xtree_rf_features, mlp_knn_features,
                                 rf_knn_features,
                                 rf_mlp_features, xtree_mlp_features, xtree_knn_features, xtree_rf_features
                                 )


def variables_importance_single_models(train_df, test_df):
    ml_utils = MachineLearningUtils()
    x_train, x_test, y_train, y_test = ml_utils.get_training_data(train_df=train_df, predicted_var='action', test_size=0.0000001)
    if test_df is not None:
        x_test, y_test = ml_utils.get_external_test(train_df=test_df, predicted_var='action')
    knn = ml_utils.knn_train(x_train, x_test, y_train, y_test)
    rf = ml_utils.random_forest_train(x_train, x_test, y_train, y_test)
    xtree = ml_utils.extra_tree_train(x_train, x_test, y_train, y_test)


    for importance_cut in [0]:
        importance_cut = importance_cut

        result = permutation_importance(knn, x_test, y_test, n_repeats=10, random_state=42)
        # Obtener importancia de características
        importance = result.importances_mean
        # Asociar importancia con nombres de características
        feature_names = train_df.columns  # Reemplaza con tus nombres de características
        feature_importance_dict = dict(zip(feature_names, importance))
        # Imprimir importancia de características

        knn_features = [feature for feature, importance in feature_importance_dict.items() if
                        importance > importance_cut]

        result = permutation_importance(rf, x_test, y_test, n_repeats=10, random_state=42)
        # Obtener importancia de características
        importance = result.importances_mean
        # Asociar importancia con nombres de características
        feature_names = train_df.columns  # Reemplaza con tus nombres de características
        feature_importance_dict = dict(zip(feature_names, importance))
        # Imprimir importancia de características

        rf_features = [feature for feature, importance in feature_importance_dict.items() if
                       importance > importance_cut]

        result = permutation_importance(xtree, x_test, y_test, n_repeats=10, random_state=42)
        # Obtener importancia de características
        importance = result.importances_mean
        # Asociar importancia con nombres de características
        feature_names = train_df.columns  # Reemplaza con tus nombres de características
        feature_importance_dict = dict(zip(feature_names, importance))
        # Imprimir importancia de características

        xtree_features = [feature for feature, importance in feature_importance_dict.items() if
                          importance > importance_cut]

        return knn_features, rf_features, xtree_features


def get_cross_variables_and():
    set_knn = set(knn_variables)
    set_mlp = set(mlp_variables)
    set_rf = set(rf_variables)
    set_xtree = set(xtree_variables)
    cross_variables = set_knn & set_mlp & set_rf & set_xtree
    return list(cross_variables)


def get_cross_variables_or():
    set_knn = set(knn_variables)
    set_mlp = set(mlp_variables)
    set_rf = set(rf_variables)
    set_xtree = set(xtree_variables)
    cross_variables = set_knn | set_mlp | set_rf | set_xtree
    return list(cross_variables)
