import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_and_save_model(data_path: str, model_output_path: str = "model.joblib"):
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    joblib.dump(clf, model_output_path)
    print(f"Model successfully saved to {model_output_path}")


if __name__ == "__main__":
    train_and_save_model("creditcard.csv")
