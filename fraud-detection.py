from sklearn.preprocessing import LabelEncoder
import polars as pl
import glob
import joblib

rf_model = joblib.load("fraud_model.pkl")
label_encoders = joblib.load("fraud_label_encoders.pkl")
# dummy data
test_transaction = pl.read_csv("ieee-fraud-detection/test_transaction_decoded.csv")
X_dummy = test_transaction.drop(['TransactionID'])

def main(X):
    y = rf_model.predict(X)
    if y == 0:
        y_ = "not fraud"
    elif y == 1:
        y_ = "fraud"
    return {"y": y_, "x":X}


if __name__ == "__main__":
    X = X_dummy[0]
    result = main(X)
    print(result)