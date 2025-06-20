from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging
import pandas as pd
import numpy as np

def train_model(df: pd.DataFrame, y: np.ndarray) -> None:
    """
    Train a RandomForestClassifier and print accuracy.
    """
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy:.4f}")
