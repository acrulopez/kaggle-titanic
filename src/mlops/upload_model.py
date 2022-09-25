from google.cloud import storage
from sklearn.pipeline import make_pipeline, Pipeline
import os, pickle


def process_gcs_uri(uri: str):
    """
    Receives a Google Cloud Storage (GCS) uri and breaks it down to the scheme, bucket, path and file

            Parameters:
                    uri (str): GCS uri

            Returns:
                    scheme (str): uri scheme
                    bucket (str): uri bucket
                    path (str): uri path
                    file (str): uri file
    """
    url_arr = uri.split("/")
    if "." not in url_arr[-1]:
        file = ""
    else:
        file = url_arr.pop()
    scheme = url_arr[0]
    bucket = url_arr[2]
    path = "/".join(url_arr[3:])
    path = path[:-1] if path.endswith("/") else path

    return scheme, bucket, path, file


def pipeline_export_gcs(fitted_pipeline: Pipeline, model_dir: str) -> str:
    """
    Exports trained pipeline to GCS

            Parameters:
                    fitted_pipeline (sklearn.pipelines.Pipeline): the Pipeline object with data already fitted (trained pipeline object)
                    model_dir (str): GCS path to store the trained pipeline. i.e gs://example_bucket/training-job
            Returns:
                    export_path (str): Model GCS location
    """
    scheme, bucket, path, file = process_gcs_uri(model_dir)
    if scheme != "gs:":
        raise ValueError("URI scheme must be gs")

    # Upload the model to GCS
    b = storage.Client().bucket(bucket)
    export_path = os.path.join(path, "model.pkl")
    blob = b.blob(export_path)

    blob.upload_from_string(pickle.dumps(fitted_pipeline))
    return scheme + "//" + os.path.join(bucket, export_path)
