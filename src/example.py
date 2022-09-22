from ingestion import ingest_data
from cleaning import feature_engineer
from utils import split_x_y
from constants import SVC_RF_FORMULA, LOGIT_FORMULA
from training import train_logit, train_random_forest, train_svc


if __name__ == "__main__":
    # Ingest the data
    df = ingest_data("data/train.csv")

    # Perform feature engineering
    df = feature_engineer(df)

    # Get the x and the y from the dataframe
    y, x = y, x = split_x_y(df, SVC_RF_FORMULA)

    # ------------------------------ Random Forest ------------------------------
    print("------------------------- Random forest -------------------------")
    logit_model = train_random_forest(df)
    predicts = logit_model.predict(x.iloc[:5])
    values = y.iloc[:5].to_numpy().flatten()
    print(predicts == values)
