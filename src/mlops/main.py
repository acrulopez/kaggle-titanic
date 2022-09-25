from ingestor import ingest_data
from cleaning import feature_engineer
from utils import split_x_y
from constants import SVC_RF_FORMULA
from training import train_random_forest
from upload_model import pipeline_export_gcs
import argparse, os


if __name__ == "__main__":

    # Get the model directory if was provided
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        help="Directory to output model and artifacts",
        type=str,
        default=os.environ["AIP_MODEL_DIR"] if "AIP_MODEL_DIR" in os.environ else "",
    )
    args = parser.parse_args()
    arguments = args.__dict__

    # Ingest the data
    print("\nIngesting data...\n")
    df = ingest_data("data/train.csv")

    # Perform feature engineering
    print("Cleaning data...\n")
    df = feature_engineer(df)

    # Get the x and the y from the dataframe
    y, x = y, x = split_x_y(df, SVC_RF_FORMULA)

    # Train the model
    print("Training random forest...\n")
    clf = train_random_forest(df)

    # Compute accuracy (log it on the real system)
    predicts = clf.predict(x)
    values = y.to_numpy().flatten()
    print("Accuracy on train set:", (predicts == values).sum() / len(predicts), "\n")

    # Upload the model
    if arguments["model_dir"] != "":
        print("Uploading model...\n")
        pipeline_export_gcs(clf, arguments["model_dir"])
    else:
        print("Not uploading model because no '--model-dir' given as argument\n")
