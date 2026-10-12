from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


DATA_PATH = Path(__file__).with_name("Iris.csv")


def load_dataset():
    """Load the Iris dataset and separate features and target."""
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Id", "Species"])
    y = df["Species"]

    return X, y


def train_model():
    """Train the classifier and return evaluation results."""
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, X_test, y_test, accuracy


if __name__ == "__main__":
    model, X_test, y_test, accuracy = train_model()

    print("Iris Classification Model")
    print("Test records:", len(y_test))
    print(f"Test accuracy: {accuracy:.2%}")
