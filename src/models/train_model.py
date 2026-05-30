from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

class ModelTrainer:
    def __init__(self, target_column, random_state=42):
        self.target_column = target_column
        self.random_state = random_state

    def train(self, df):
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return {"model": model, "X_test": X_test, "y_test": y_test, "preds": preds}

def save_model(model, path):
    import joblib
    joblib.dump(model, path)