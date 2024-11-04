from kfp.dsl import component, pipeline
import kfp
from kfp import kubernetes


@component(
    packages_to_install=["pandas", "numpy", "scikit-learn", "s3fs"],
    base_image="python:3.9",
)
def prepare_data(
    data_path: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_default_region: str,
    aws_s3_endpoint_url: str,
):
    import pandas as pd
    import os
    from sklearn import datasets

    os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    os.environ["AWS_DEFAULT_REGION"] = aws_default_region

    df = pd.read_csv(
        "s3://data/iris.csv",
        storage_options={
            "key": aws_access_key_id,
            "secret": aws_secret_access_key,
            "token": None,
            "client_kwargs": {
                "endpoint_url": f"http://{aws_s3_endpoint_url}",
            },
        },
    )

    df = df.dropna()
    df.to_csv(f"{data_path}/final_df.csv", index=False)


@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.9",
)
def train_test_split(data_path: str, target_column: str = "variety"):
    import pandas as pd
    import numpy as np
    import os
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(f"{data_path}/final_df.csv")

    # Separate features and target from the DataFrame
    X = df.drop(target_column, axis=1)  # Features
    y = df[target_column]  # Target

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    np.save(f"{data_path}/X_train.npy", X_train)
    np.save(f"{data_path}/X_test.npy", X_test)
    np.save(f"{data_path}/y_train.npy", y_train)
    np.save(f"{data_path}/y_test.npy", y_test)


@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.9",
)
def training_basic_classifier(data_path: str):
    import pandas as pd
    import json
    import numpy as np
    import os
    import random
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    X_train = np.load(f"{data_path}/X_train.npy", allow_pickle=True)
    y_train = np.load(f"{data_path}/y_train.npy", allow_pickle=True)
    X_test = np.load(f"{data_path}/X_test.npy", allow_pickle=True)

    hyperparameters = {
        "n_estimators": random.randint(1, 2),  # Number of trees in the forest
        "max_depth": random.randint(1, 5),
        # Maximum depth of the tree (None means nodes are expanded until all leaves are pure)
        "min_samples_split": random.randint(
            2, 3
        ),  # Minimum number of samples required to split an internal node
        "min_samples_leaf": random.randint(
            1, 2
        ),  # Minimum number of samples required to be at a leaf node
        "random_state": 42,  # Controls the randomness of the estimator for reproducibility
    }
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # overwrite X_test with scaled values
    np.save(f"{data_path}/X_test.npy", X_test)

    # Create and train the Random Forest Classifier model with hyperparameters
    model = RandomForestClassifier(**hyperparameters)
    model.fit(X_train, y_train)
    import pickle

    with open(f"{data_path}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    # hyperparameters to file
    with open(f"{data_path}/hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f)


@component(
    packages_to_install=["pandas", "numpy", "scikit-learn", "mlflow", "boto3"],
    base_image="python:3.9",
)
def register_model(
    data_path: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_default_region: str,
    aws_s3_endpoint_url: str,
    # training_meta: str,
) -> dict:
    import pandas as pd
    import numpy as np
    import pickle
    import os
    import mlflow
    import random
    import json
    from mlflow.models import infer_signature

    with open(f"{data_path}/model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    # hyperparameters = json.loads(training_meta["hyperparams"])

    # read hyperparameters from file
    with open(f"{data_path}/hyperparameters.json", "r") as f:
        hyperparameters = json.load(f)

    # Infer the model signature
    X_test = np.load(f"{data_path}/X_test.npy", allow_pickle=True)
    y_pred = rf_model.predict(X_test)
    signature = infer_signature(X_test, y_pred)

    # Set AWS credentials in the environment
    os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    os.environ["AWS_DEFAULT_REGION"] = aws_default_region
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{aws_s3_endpoint_url}"

    # log and register the model using MLflow scikit-learn API
    mlflow.set_tracking_uri("http://mlflow-svc.mlflow:5000")
    reg_model_name = "SklearnIrisRFModel"

    exp_name = "iris-experiment"
    existing_exp = mlflow.get_experiment_by_name(exp_name)
    if not existing_exp:
        mlflow.create_experiment(exp_name)
    mlflow.set_experiment(exp_name)

    with mlflow.start_run() as run:
        mlflow.log_params(hyperparameters)

        # Log model artifact to S3
        artifact_path = "sklearn-model"
        mlflow.log_artifact(
            local_path=f"{data_path}/model.pkl", artifact_path=artifact_path
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=rf_model,
            artifact_path="sklearn-model",
            signature=signature,
            registered_model_name=reg_model_name,
        )

    model_uri = f"runs:/{run.info.run_id}/sklearn-model"

    # Register model linked to S3 artifact location
    mlflow.register_model(model_uri, reg_model_name)

    return {
        "artifact_path": artifact_path,
        "artifact_uri": run.info.artifact_uri,
        "run_id": run.info.run_id,
    }


@component(
    packages_to_install=["pandas", "numpy", "scikit-learn", "mlflow", "boto3"],
    base_image="python:3.9",
)
def predict_on_test_data(
    data_path: str,
    model_info: dict,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_default_region: str,
    aws_s3_endpoint_url: str,
) -> str:
    import pandas as pd
    from sklearn.metrics import accuracy_score
    import numpy as np
    import pickle
    import os
    import mlflow

    # Set AWS credentials in the environment
    os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    os.environ["AWS_DEFAULT_REGION"] = aws_default_region
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{aws_s3_endpoint_url}"

    artifact_path = model_info["artifact_path"]
    artifact_uri = model_info["artifact_uri"]
    run_id = model_info["run_id"]

    mlflow.set_tracking_uri("http://mlflow-svc.mlflow:5000")
    model_uri = f"{artifact_uri}/{artifact_path}"
    rf_model = mlflow.sklearn.load_model(model_uri)

    X_test = np.load(f"{data_path}/X_test.npy", allow_pickle=True)
    y_pred = rf_model.predict(X_test)
    y_test = np.load(f"{data_path}/y_test.npy", allow_pickle=True)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    # Log the accuracy metric to the MLflow run
    with mlflow.start_run(run_id=run_id) as run:
        mlflow.log_metric("accuracy", accuracy)
    return model_uri


@component(
    packages_to_install=["kserve"],
    base_image="python:3.9",
)
def model_serving(model_uri: str):
    from kubernetes import client
    from kserve import KServeClient
    from kserve import constants
    from kserve import utils
    from kserve import V1beta1InferenceService
    from kserve import V1beta1InferenceServiceSpec
    from kserve import V1beta1PredictorSpec
    from kserve import V1beta1SKLearnSpec
    import os
    import time

    namespace = utils.get_default_target_namespace()

    name = "iris-rf"
    kserve_version = "v1beta1"
    api_version = constants.KSERVE_GROUP + "/" + kserve_version
    # create InferenceService
    isvc = V1beta1InferenceService(
        api_version=api_version,
        kind=constants.KSERVE_KIND,
        metadata=client.V1ObjectMeta(
            name=name,
            namespace=namespace,
            annotations={"sidecar.istio.io/inject": "false"},
        ),
        spec=V1beta1InferenceServiceSpec(
            predictor=V1beta1PredictorSpec(
                service_account_name="mlflow-sa",
                sklearn=(
                    V1beta1SKLearnSpec(
                        storage_uri=model_uri,
                        resources=client.V1ResourceRequirements(
                            limits={"cpu": "500m", "memory": "2Gi"},
                            requests={"cpu": "250m", "memory": "1Gi"},
                        ),
                    )
                ),
            ),
        ),
    )

    kserve = KServeClient()
    try:
        kserve.delete(name=name, namespace=namespace)
        print("Deleted existing InferenceService")
    except:
        print("No existing InferenceService found")
    time.sleep(10)

    kserve.create(isvc)


@component(
    packages_to_install=["requests", "mlflow"],
    base_image="python:3.9",
)
def send_slack_message(model_info: dict):
    import requests
    import mlflow
    import json

    run_id = model_info["run_id"]
    mlflow.set_tracking_uri("http://mlflow-svc.mlflow:5000")
    mlflow_client = mlflow.MlflowClient()
    run_data_dict = mlflow_client.get_run(run_id).data.to_dictionary()
    accuracy = run_data_dict["metrics"]["accuracy"]
    accuracy = round(accuracy, 2)
    # Send a slack notification if the accuracy is greater than 0.95
    if accuracy > 0.95:
        title = f"Threshold Reached"
        message = f"Model with run_id {run_id} has reached accuracy threshold of 0.95 with current acc: {accuracy}. %"
        slack_data = {
            "username": "MachineLearningAlerts",
            "icon_emoji": ":robot_face:",
            "channel": "#general",
            "attachments": [
                {
                    "color": "#9733EE",
                    "fields": [
                        {
                            "title": title,
                            "value": message,
                            "short": "false",
                        }
                    ],
                }
            ],
        }

        headers = {
            "Content-Type": "application/json",
        }
        webhook_url = "https://hooks.slack.com/services/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"
        response = requests.post(
            webhook_url, data=json.dumps(slack_data), headers=headers
        )
        if response.status_code == 200:
            print("Notification Sent....")


from kubernetes import client, config
import base64


@pipeline(
    name="iris-pipeline",
)
def iris_pipeline(data_path: str):
    pvc1 = kubernetes.CreatePVC(
        # can also use pvc_name instead of pvc_name_suffix to use a pre-existing PVC
        pvc_name_suffix="-iris-mlflow-pvc",
        access_modes=["ReadWriteMany"],
        size="1Mi",
        storage_class_name="standard",
    )

    # Load Kubernetes configuration
    config.load_kube_config()

    # Fetch the AWS credentials from the secret
    secret_name = "miniocreds"
    secret_namespace = "kubeflow"
    secret_key_id = "AWS_ACCESS_KEY_ID"
    secret_key_access = "AWS_SECRET_ACCESS_KEY"
    secret_region = "AWS_DEFAULT_REGION"

    v1 = client.CoreV1Api()
    secret = v1.read_namespaced_secret(secret_name, namespace=secret_namespace)

    # Convert bytes to string
    aws_access_key_id = base64.b64decode(secret.data[secret_key_id]).decode("utf-8")
    aws_secret_access_key = base64.b64decode(secret.data[secret_key_access]).decode(
        "utf-8"
    )
    aws_default_region = base64.b64decode(secret.data[secret_region]).decode("utf-8")
    aws_s3_endpoint_url = secret.metadata.annotations["serving.kserve.io/s3-endpoint"]

    prepare_data_task = prepare_data(
        data_path=data_path,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_default_region=aws_default_region,
        aws_s3_endpoint_url=aws_s3_endpoint_url,
    ).set_retry(3)
    prepare_data_task.set_caching_options(False)
    kubernetes.mount_pvc(
        prepare_data_task, pvc_name=pvc1.outputs["name"], mount_path="/data"
    )

    train_test_split_task = train_test_split(data_path=data_path).set_retry(3)
    train_test_split_task.set_caching_options(False)

    kubernetes.mount_pvc(
        train_test_split_task, pvc_name=pvc1.outputs["name"], mount_path="/data"
    )
    train_test_split_task.after(prepare_data_task)

    training_basic_classifier_task = training_basic_classifier(
        data_path=data_path
    ).set_retry(3)
    kubernetes.mount_pvc(
        training_basic_classifier_task,
        pvc_name=pvc1.outputs["name"],
        mount_path="/data",
    )
    training_basic_classifier_task.set_caching_options(False)
    training_basic_classifier_task.after(train_test_split_task)

    register_model_task = register_model(
        data_path=data_path,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_default_region=aws_default_region,
        aws_s3_endpoint_url=aws_s3_endpoint_url,
    ).set_retry(3)
    register_model_task.set_caching_options(False)
    kubernetes.mount_pvc(
        register_model_task, pvc_name=pvc1.outputs["name"], mount_path="/data"
    )
    register_model_task.after(training_basic_classifier_task)

    predict_on_test_data_task = predict_on_test_data(
        data_path=data_path,
        model_info=register_model_task.output,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_default_region=aws_default_region,
        aws_s3_endpoint_url=aws_s3_endpoint_url,
    ).set_retry(3)
    predict_on_test_data_task.set_caching_options(False)
    kubernetes.mount_pvc(
        predict_on_test_data_task, pvc_name=pvc1.outputs["name"], mount_path="/data"
    )
    predict_on_test_data_task.after(register_model_task)

    model_serving_task = model_serving(
        model_uri=predict_on_test_data_task.output
    ).set_retry(3)
    model_serving_task.after(predict_on_test_data_task)

    send_slack_message_task = send_slack_message(
        model_info=register_model_task.output
    ).set_retry(3)
    send_slack_message_task.after(model_serving_task)

    delete_pvc1 = kubernetes.DeletePVC(pvc_name=pvc1.outputs["name"]).after(
        send_slack_message_task
    )


from kfp import compiler  # noqa: F811

compiler.Compiler().compile(
    pipeline_func=iris_pipeline, package_path="iris_mlflow_kserve_pipeline.yaml"
)