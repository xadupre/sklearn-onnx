# SPDX-License-Identifier: Apache-2.0
import unittest
import packaging.version as pv
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from onnxruntime import __version__ as ort_version


class TestInvestigate(unittest.TestCase):
    def test_issue_1053(self):
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        import onnxruntime as rt
        from skl2onnx.common.data_types import FloatTensorType
        from skl2onnx import convert_sklearn

        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # Fitting logistic regression model.
        for cls in [LogisticRegression, DecisionTreeClassifier]:
            with self.subTest(cls=cls):
                clr = cls()  # Use logistic regression instead of decision tree.
                clr.fit(X_train, y_train)

                initial_type = [
                    ("float_input", FloatTensorType([None, 4]))
                ]  # Remove the batch dimension.
                onx = convert_sklearn(clr, initial_types=initial_type, target_opset=12)

                sess = rt.InferenceSession(
                    onx.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                input_name = sess.get_inputs()[0].name
                label_name = sess.get_outputs()[0].name
                pred_onx = sess.run(
                    [label_name], {input_name: X_test[:1].astype("float32")}
                )[
                    0
                ]  # Select a single sample.
                self.assertEqual(len(pred_onx.tolist()), 1)

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.16.0"),
        reason="opset 19 not implemented",
    )
    @ignore_warnings(category=(ConvergenceWarning,))
    def test_issue_1055(self):
        import numpy as np
        from numpy.testing import assert_almost_equal
        import sklearn.feature_extraction.text
        import sklearn.linear_model
        import sklearn.pipeline
        import onnxruntime as rt
        import skl2onnx.common.data_types

        lr = sklearn.linear_model.LogisticRegression(
            C=100,
            multi_class="multinomial",
            solver="sag",
            class_weight="balanced",
            n_jobs=-1,
        )
        tf = sklearn.feature_extraction.text.TfidfVectorizer(
            token_pattern="\\w+|[^\\w\\s]+",
            ngram_range=(1, 1),
            max_df=1.0,
            min_df=1,
            sublinear_tf=True,
        )

        pipe = sklearn.pipeline.Pipeline([("transformer", tf), ("logreg", lr)])

        corpus = [
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
            "more text",
            "$words",
            "I keep writing things",
            "how many documents now?",
            "this is a really long setence",
            "is this a final document?",
        ]
        labels = ["1", "2", "1", "2", "1", "2", "1", "2", "1", "2"]

        pipe.fit(corpus, labels)

        onx = skl2onnx.convert_sklearn(
            pipe,
            "a model",
            initial_types=[
                ("input", skl2onnx.common.data_types.StringTensorType([None, 1]))
            ],
            target_opset=19,
            options={"zipmap": False},
        )
        for d in onx.opset_import:
            if d.domain == "":
                self.assertEqual(d.version, 19)
            elif d.domain == "com.microsoft":
                self.assertEqual(d.version, 1)
            elif d.domain == "ai.onnx.ml":
                self.assertEqual(d.version, 1)

        expected = pipe.predict_proba(corpus)
        sess = rt.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        got = sess.run(None, {"input": np.array(corpus).reshape((-1, 1))})
        assert_almost_equal(expected, got[1], decimal=2)

    @unittest.skipIf(
        pv.Version(ort_version) < pv.Version("1.16.0"),
        reason="opset 19 not implemented",
    )
    def test_issue_1069(self):
        import math
        from typing import Any
        import numpy
        import pandas
        from sklearn import (
            base,
            compose,
            ensemble,
            linear_model,
            pipeline,
            preprocessing,
            datasets,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        import onnxruntime
        from skl2onnx import to_onnx
        from skl2onnx.sklapi import CastTransformer

        class FLAGS:
            classes = 7
            samples = 1000
            timesteps = 5
            trajectories = int(1000 / 5)
            features = 10
            seed = 10

        columns = [
            f"facelandmark{i}" for i in range(1, int(FLAGS.features / 2) + 1)
        ] + [f"poselandmark{i}" for i in range(1, int(FLAGS.features / 2) + 1)]

        X, y = datasets.make_classification(
            n_classes=FLAGS.classes,
            n_informative=math.ceil(math.log2(FLAGS.classes * 2)),
            n_samples=FLAGS.samples,
            n_features=FLAGS.features,
            random_state=FLAGS.seed,
        )

        X = pandas.DataFrame(X, columns=columns)

        X["trajectory"] = numpy.repeat(
            numpy.arange(FLAGS.trajectories), FLAGS.timesteps
        )
        X["timestep"] = numpy.tile(numpy.arange(FLAGS.timesteps), FLAGS.trajectories)

        trajectory_train, trajectory_test = train_test_split(
            X["trajectory"].unique(),
            test_size=0.25,
            random_state=FLAGS.seed,
        )

        trajectory_train, trajectory_test = set(trajectory_train), set(trajectory_test)

        X_train, X_test = (
            X[X["trajectory"].isin(trajectory_train)],
            X[X["trajectory"].isin(trajectory_test)],
        )
        y_train, _ = y[X_train.index], y[X_test.index]

        def augment_with_lag_timesteps(X, k, columns):
            augmented = X.copy()

            for i in range(1, k + 1):
                shifted = X[columns].groupby(X["trajectory"]).shift(i)
                shifted.columns = [f"{x}_lag{i}" for x in shifted.columns]

                augmented = pandas.concat([augmented, shifted], axis=1)

            return augmented

        X_train = augment_with_lag_timesteps(X_train, k=3, columns=X.columns[:-2])
        X_test = augment_with_lag_timesteps(X_test, k=3, columns=X.columns[:-2])

        X_train.drop(columns=["trajectory", "timestep"], inplace=True)
        X_test.drop(columns=["trajectory", "timestep"], inplace=True)

        def abc_Embedder() -> list[tuple[str, Any]]:
            return [
                ("cast64", CastTransformer(dtype=numpy.float64)),
                ("scaler", preprocessing.StandardScaler()),
                ("cast32", CastTransformer()),
                ("basemodel", DecisionTreeClassifier(max_depth=2)),
            ]

        def Classifier(features: list[str]) -> base.BaseEstimator:
            feats = [i for i, x in enumerate(features) if x.startswith("facelandmark")]

            classifier = ensemble.StackingClassifier(
                estimators=[
                    (
                        "facepipeline",
                        pipeline.Pipeline(
                            [
                                (
                                    "preprocessor",
                                    compose.ColumnTransformer(
                                        [("identity", "passthrough", feats)]
                                    ),
                                ),
                                ("embedder", pipeline.Pipeline(steps=abc_Embedder())),
                            ]
                        ),
                    ),
                    (
                        "posepipeline",
                        pipeline.Pipeline(
                            [
                                (
                                    "preprocessor",
                                    compose.ColumnTransformer(
                                        [("identity", "passthrough", feats)]
                                    ),
                                ),
                                ("embedder", pipeline.Pipeline(steps=abc_Embedder())),
                            ]
                        ),
                    ),
                ],
                final_estimator=linear_model.LogisticRegression(
                    multi_class="multinomial"
                ),
            )

            return classifier

        model = Classifier(list(X_train.columns))
        model.fit(X_train, y_train)

        sample = X_train[:1].astype(numpy.float32)

        for m in [model.estimators_[0].steps[0][-1], model.estimators_[0], model]:
            with self.subTest(model=type(m)):
                exported = to_onnx(
                    model,
                    X=numpy.asarray(sample),
                    name="classifier",
                    target_opset={"": 12, "ai.onnx.ml": 2},
                    options={id(model): {"zipmap": False}},
                )

                modelengine = onnxruntime.InferenceSession(
                    exported.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                assert modelengine is not None

    def test_issue_1086(self):

        # import pandas as pd
        import numpy as np
        import lightgbm as lgb
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import load_iris
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import (
            accuracy_score,
            roc_auc_score,
            classification_report,
        )
        from sklearn.base import BaseEstimator, TransformerMixin
        import joblib

        from skl2onnx.common.data_types import FloatTensorType
        from skl2onnx import convert_sklearn, update_registered_converter
        from skl2onnx.common.shape_calculator import (
            calculate_linear_classifier_output_shapes,
        )
        from skl2onnx.proto import onnx_proto

        data = load_iris(as_frame=True)
        df = data["frame"]
        df.rename(
            {
                "sepal length (cm)": "sepal_length",
                "sepal width (cm)": "sepal_width",
                "petal length (cm)": "petal_length",
                "petal width (cm)": "petal_width",
            },
            axis=1,
            inplace=True,
        )

        def create_target_column(df, target):
            return np.where(df["target"] == target, 1, 0)

        # target_dict = {
        #     0 : 'setosa',
        #     1 : 'versicolor',
        #     2 : 'virginica'
        # }

        df["is_setosa"] = create_target_column(df, 0)
        df["is_versicolor"] = create_target_column(df, 1)
        df["is_virginica"] = create_target_column(df, 2)

        features = df.columns[:-4]

        class TrainModel(object):
            def __init__(
                self, X_train, X_test, X_valid, y_train, y_test, y_valid, target_name
            ):
                self.X_train = X_train
                self.X_test = X_test
                self.y_train = y_train
                self.y_test = y_test
                self.X_valid = X_valid
                self.y_valid = y_valid
                self.target_name = target_name

            def train_rf_model(self):
                rf = RandomForestClassifier(n_estimators=100, random_state=0)  # seed)
                rf.fit(self.X_train, self.y_train)
                return rf

            def train_lgbm_model(self):
                train = lgb.Dataset(self.X_train, label=self.y_train)
                valid_sets = [(self.X_test, self.y_test)]
                params = {}
                params["random_state"] = 0  # seed
                params["n_estimators"] = 100
                params["learning_rate"] = 0.001
                params["boosting_type"] = "gbdt"
                params["objective"] = "binary"
                params["metric"] = {"binary_logloss", "auc"}
                params["tree_learner"] = "data"
                # training the model
                res = {}
                clf = lgb.train(
                    params,
                    train,
                    valid_sets,
                    valid_names=["valid"],
                    evals_result=res,
                    callbacks=[
                        lgb.reset_parameter(learning_rate=lambda x: 0.95**x * 0.1)
                    ],
                )
                return clf

            def predict(self, model, df=None):
                if df is not None:
                    data = df
                else:
                    data = self.X_valid
                y_probs = model.predict(data)
                y_pred = (y_probs > 0.5).astype("int")
                return y_probs, y_pred

            def compute_metrics(self, y_pred, y_truth=None):
                if y_truth is not None:
                    pass
                else:
                    y_truth = self.y_valid
                metrics = dict()
                metrics["lob"] = self.target_name
                metrics["classification_report"] = classification_report(
                    y_truth, y_pred
                )
                metrics["roc_auc_score"] = roc_auc_score(y_truth, y_pred)
                metrics["accuracy_score"] = accuracy_score(y_truth, y_pred)
                return metrics

            def print_metrics(self, metrics):
                print(
                    "classification_report:\n", metrics["classification_report"], "\n\n"
                )
                print("roc_auc_score:\n", metrics["roc_auc_score"], "\n\n")
                print("accuracy_score:\n", metrics["accuracy_score"], "\n\n")

            def train(self, model_name="lgbm"):
                if model_name == "lgbm":
                    model = self.train_lgbm_model()
                elif model_name == "rf":
                    model = self.train_rf_model()
                else:
                    return "Supported model keywords are: [LightGBM: 'lgbm', RandomForest: 'rf']"
                y_logs, y_pred = self.predict(model)
                metrics = self.compute_metrics(y_pred)
                return model, metrics, y_pred, y_logs

        def train_model_per_target(target_name, model_name="lgbm"):
            X_train, X_test, y_train, y_test = train_test_split(
                df[features], df[f"is_{target_name}"], test_size=0.2
            )
            tm = TrainModel(
                X_train, X_test, X_test, y_train, y_test, y_test, target_name
            )
            model, _, _, _ = tm.train(model_name)
            return model, X_train, X_test, y_train, y_test

        setosa_model, setosa_X_train, setosa_X_test, setosa_y_train, setosa_y_test = (
            train_model_per_target(target_name="setosa")
        )

        (
            versicolor_model,
            versicolor_X_train,
            versicolor_X_test,
            versicolor_y_train,
            versicolor_y_test,
        ) = train_model_per_target(target_name="versicolor")

        (
            virginica_model,
            virginica_X_train,
            virginica_X_test,
            virginica_y_train,
            virginica_y_test,
        ) = train_model_per_target(target_name="virginica", model_name="rf")

        class SetosaPredictionModel(BaseEstimator, TransformerMixin):
            def __init__(self):
                self.features = [
                    "sepal_length",
                    "sepal_width",
                    "petal_length",
                    "petal_width",
                ]
                self.model = lgb.Booster(
                    model_file="/Workspace/Users/[Achyuta.Jha@amexgbt.com/vta_poc_models/setosa_lgbm_model.txt](http://Achyuta.Jha@amexgbt.com/vta_poc_models/setosa_lgbm_model.txt)"
                )

            def fit(self, X, y=None):
                return None

            def transform(self, X):
                y_probs = self.model.predict(X[self.features])
                X["setosa_pred"] = (y_probs > 0.5).astype("int")
                return X

        class VersicolorPredictionModel(BaseEstimator, TransformerMixin):
            def __init__(self):
                self.features = [
                    "sepal_length",
                    "sepal_width",
                    "petal_length",
                    "petal_width",
                ]
                self.model = lgb.Booster(
                    model_file="/Workspace/Users/[Achyuta.Jha@amexgbt.com/vta_poc_models/versicolor_lgbm_model.txt](http://Achyuta.Jha@amexgbt.com/vta_poc_models/versicolor_lgbm_model.txt)"
                )

            def fit(self, X, y=None):
                return None

            def transform(self, X):
                y_probs = self.model.predict(X[self.features])
                X["versicolor_pred"] = (y_probs > 0.5).astype("int")
                return X

        class VirginicaPredictionModel(BaseEstimator, TransformerMixin):
            def __init__(self):
                self.features = [
                    "sepal_length",
                    "sepal_width",
                    "petal_length",
                    "petal_width",
                ]
                self.model = joblib.load(
                    "/Workspace/Users/[Achyuta.Jha@amexgbt.com/vta_poc_models/virginica_rf_model.joblib](http://Achyuta.Jha@amexgbt.com/vta_poc_models/virginica_rf_model.joblib)"
                )

            def fit(self, X, y=None):
                return None

            def predict(self, X):
                pred_list = []
                pred_list.append(X["setosa_pred"].values)
                pred_list.append(X["versicolor_pred"].values)
                pred_list.append(self.model.predict(X[self.features]))
                return pred_list

        pipeline = Pipeline(
            [
                ("setosa_prediction_model", SetosaPredictionModel()),
                ("versicolor_prediction_model", VersicolorPredictionModel()),
                ("virginica_prediction_model", VirginicaPredictionModel()),
            ]
        )

        def convert_setosa_model(scope, operator, container):
            # op = operator.raw_operator
            inputs = operator.inputs
            outputs = operator.outputs
            name = scope.get_unique_operator_name("SetosaModel")

            # feature_names = op.features
            input_names = [inputs[i].full_name for i in range(len(inputs))]
            input_name = input_names[0]

            probabilities_name = scope.get_unique_variable_name(name + "_probabilities")
            probability_name = scope.get_unique_variable_name(name + "_probability")

            container.add_node(
                "LgbmPredict",
                input_name,
                probabilities_name,
                op_domain="[ai.onnx.ml](http://ai.onnx.ml/)",
                name=name,
            )

            # probabilities_shape = (1, 2)
            container.add_node(
                "ArrayFeatureExtractor",
                probabilities_name,
                probability_name,
                op_domain="[ai.onnx.ml](http://ai.onnx.ml/)",
                name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
                attr={"indices": [1]},
            )

            container.add_node(
                "GreaterOrEqual",
                probability_name,
                outputs[0].full_name,
                op_domain="[ai.onnx.ml](http://ai.onnx.ml/)",
                name=scope.get_unique_operator_name("GreaterOrEqual"),
                attr={},
            )
            container.add_node(
                "Cast",
                outputs[0].full_name,
                outputs[0].full_name,
                op_domain="[ai.onnx.ml](http://ai.onnx.ml/)",
                name=scope.get_unique_operator_name("Cast"),
                attr={"to": onnx_proto.TensorProto.INT64},
            )

            # Set the number of classes
            operator.target_opset[-1].set_onnx_attr("n_classes", 2)

        update_registered_converter(
            SetosaPredictionModel,
            "SetosaLGBMClassifier",
            calculate_linear_classifier_output_shapes,
            convert_setosa_model,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )

        def convert_versicolor_model(scope, operator, container):
            # op = operator.raw_operator
            inputs = operator.inputs
            outputs = operator.outputs
            name = scope.get_unique_operator_name("SetosaModel")

            # feature_names = op.features
            input_names = [inputs[i].full_name for i in range(len(inputs))]
            input_name = input_names[0]

            probabilities_name = scope.get_unique_variable_name(name + "_probabilities")
            probability_name = scope.get_unique_variable_name(name + "_probability")

            container.add_node(
                "LgbmPredict",
                input_name,
                probabilities_name,
                op_domain="[ai.onnx.ml](http://ai.onnx.ml/)",
                name=name,
            )

            # probabilities_shape = (1, 2)
            container.add_node(
                "ArrayFeatureExtractor",
                probabilities_name,
                probability_name,
                op_domain="[ai.onnx.ml](http://ai.onnx.ml/)",
                name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
                attr={"indices": [1]},
            )

            container.add_node(
                "GreaterOrEqual",
                probability_name,
                outputs[0].full_name,
                op_domain="[ai.onnx.ml](http://ai.onnx.ml/)",
                name=scope.get_unique_operator_name("GreaterOrEqual"),
                attr={},
            )
            container.add_node(
                "Cast",
                outputs[0].full_name,
                outputs[0].full_name,
                op_domain="[ai.onnx.ml](http://ai.onnx.ml/)",
                name=scope.get_unique_operator_name("Cast"),
                attr={"to": onnx_proto.TensorProto.INT64},
            )

            # Set the number of classes
            operator.target_opset[-1].set_onnx_attr("n_classes", 2)

        update_registered_converter(
            VersicolorPredictionModel,
            "VersicolorLGBMClassifier",
            calculate_linear_classifier_output_shapes,
            convert_versicolor_model,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )

        def convert_virginica_model(scope, operator, container):
            # op = operator.raw_operator
            inputs = operator.inputs
            outputs = operator.outputs
            name = scope.get_unique_operator_name("VirginicaModel")

            # feature_names = op.features
            input_names = [inputs[i].full_name for i in range(len(inputs))]
            input_name = input_names[0]

            output_name = scope.get_unique_variable_name("output")
            # indices_name = scope.get_unique_variable_name("indices")

            # Convert input to a float tensor
            container.add_node(
                "Cast",
                input_name,
                input_name + "_casted",
                op_version=9,
                to=onnx_proto.TensorProto.FLOAT,
            )

            # Extract relevant features
            container.add_node(
                "ArrayFeatureExtractor",
                input_name + "_casted",
                input_name + "_extracted",
                op_domain="[ai.onnx.ml](http://ai.onnx.ml/)",
                name=scope.get_unique_operator_name("ArrayFeatureExtractor"),
                attr={"indices": [0, 1, 2, 3]},
            )

            # Reshape to a single row
            container.add_node(
                "Reshape",
                input_name + "_extracted",
                input_name + "_reshaped",
                op_domain="[ai.onnx.ml](http://ai.onnx.ml/)",
                name=scope.get_unique_operator_name("Reshape"),
                attr={"shape": {"dims": [1, 4]}},
            )

            # Convert to tensor
            container.add_node(
                "Identity",
                input_name + "_reshaped",
                output_name,
                op_domain="[ai.onnx.ml](http://ai.onnx.ml/)",
                name=scope.get_unique_operator_name("Identity"),
            )

            # Perform prediction
            container.add_node(
                "SklToOnnxLinearClassifier",
                output_name,
                outputs[0].full_name,
                op_domain="[ai.onnx.ml](http://ai.onnx.ml/)",
                name=name,
                **calculate_linear_classifier_output_shapes(
                    operator.raw_operator_, operator.target_opset
                ),
            )

            # Set the number of classes
            operator.target_opset[-1].set_onnx_attr("n_classes", 2)

        update_registered_converter(
            VirginicaPredictionModel,
            "VirginicaRFClassifier",
            calculate_linear_classifier_output_shapes,
            convert_virginica_model,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )

        model_onnx = convert_sklearn(
            pipeline,
            "pipeline_lgbrf",
            [("input", FloatTensorType([None, 4]))],
            target_opset={"": 12, "ai.onnx.ml": 2},
        )
        assert model_onnx is not None


if __name__ == "__main__":
    unittest.main(verbosity=2)
