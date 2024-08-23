import sys


def sklearn_convertors():
    """Collect convertors for Scikit-learn objects."""
    if "sklearn" in sys.modules:
        from .sklearn import (
            add_centroid_cluster_constr,
            add_decision_tree_classifier_constr,
            add_decision_tree_regressor_constr,
            add_gradient_boosting_classifier_constr,
            add_gradient_boosting_regressor_constr,
            add_linear_regression_constr,
            add_logistic_regression_constr,
            add_mlp_classifier_constr,
            add_mlp_regressor_constr,
            add_multi_output_classifier_constr,
            add_multi_output_regressor_constr,
            add_pipeline_constr,
            add_pls_regression_constr,
            add_random_forest_classifier_constr,
            add_random_forest_regressor_constr,
            add_support_vector_classifier_constr,
            add_support_vector_regressor_constr,
        )

        return {
            "LinearRegression": add_linear_regression_constr,
            "Ridge": add_linear_regression_constr,
            "Lasso": add_linear_regression_constr,
            "ElasticNet": add_linear_regression_constr,
            "LogisticRegression": add_logistic_regression_constr,
            "DecisionTreeClassifier": add_decision_tree_classifier_constr,
            "DecisionTreeRegressor": add_decision_tree_regressor_constr,
            "GradientBoostingRegressor": add_gradient_boosting_regressor_constr,
            "GradientBoostingClassifier": add_gradient_boosting_classifier_constr,
            "RandomForestRegressor": add_random_forest_regressor_constr,
            "RandomForestClassifier": add_random_forest_classifier_constr,
            "MLPRegressor": add_mlp_regressor_constr,
            "MLPClassifier": add_mlp_classifier_constr,
            "PLSRegression": add_pls_regression_constr,
            "PLSCanonical": add_pls_regression_constr,
            "LinearSVR": add_support_vector_regressor_constr,
            "SVR": add_support_vector_regressor_constr,
            "LinearSVC": add_support_vector_classifier_constr,
            "SVC": add_support_vector_classifier_constr,
            "KMeans": add_centroid_cluster_constr,
            "MiniBatchKMeans": add_centroid_cluster_constr,
            "Pipeline": add_pipeline_constr,
            "MultiOutputClassifier": add_multi_output_classifier_constr,
            "MultiOutputRegressor": add_multi_output_regressor_constr,
        }

    return {}


def pytorch_convertors():
    """Collect known PyTorch objects that can be formulated and the conversion class."""
    if "torch" in sys.modules:
        import torch  # pylint: disable=import-outside-toplevel

        from .torch import (  # pylint: disable=import-outside-toplevel
            add_sequential_constr,
        )

        return {torch.nn.Sequential: add_sequential_constr}
    return {}


def xgboost_convertors():
    """Collect known XGBoost objects that can be formulated and the conversion class."""
    if "xgboost" in sys.modules:
        import xgboost as xgb  # pylint: disable=import-outside-toplevel

        from .xgboost import (  # pylint: disable=import-outside-toplevel
            add_xgbclassifier_constr,
            add_xgbclassifier_rf_constr,
            add_xgbregressor_constr,
            add_xgbregressor_rf_constr,
        )

        return {
            xgb.XGBRegressor: add_xgbregressor_constr,
            xgb.XGBClassifier: add_xgbclassifier_constr,
            xgb.XGBRFRegressor: add_xgbregressor_rf_constr,
            xgb.XGBRFClassifier: add_xgbclassifier_rf_constr,
        }
    return {}


def lightgbm_convertors():
    """Collect known LightGBM objects that can be formulated and the conversion class."""
    if "lightgbm" in sys.modules:
        import lightgbm as lgb  # pylint: disable=import-outside-toplevel

        from .lightgbm import (  # pylint: disable=import-outside-toplevel
            add_lgbclassifier_constr,
            add_lgbregressor_constr,
        )

        return {
            lgb.LGBMRegressor: add_lgbregressor_constr,
            lgb.LGBMClassifier: add_lgbclassifier_constr,
        }
    return {}


def keras_convertors():
    """Collect known Keras objects that can be formulated and the conversion class."""
    if "tensorflow" in sys.modules:
        from tensorflow import keras  # pylint: disable=import-outside-toplevel

        from .keras import add_keras_constr  # pylint: disable=import-outside-toplevel

        return {
            keras.Sequential: add_keras_constr,
            keras.Model: add_keras_constr,
        }
    return {}


def onnx_convertors():
    """Collect known ONNX objects that can be formulated and the conversion class."""
    if "onnx" in sys.modules:
        import onnx  # pylint: disable=import-outside-toplevel

        from .onnx import add_onnx_constr  # pylint: disable=import-outside-toplevel

        return {
            onnx.ModelProto: add_onnx_constr,
            onnx.onnx_ml_pb2.ModelProto: add_onnx_constr,
        }

    return {}


def registered_predictors():
    """Return the list of registered predictors."""
    convertors = {
        **sklearn_convertors(),
        **pytorch_convertors(),
        **xgboost_convertors(),
        **lightgbm_convertors(),
        **keras_convertors(),
        **onnx_convertors(),
    }
    return convertors
