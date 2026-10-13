

import joblib
from pathlib import Path

def save_model(model):
    model_path = Path("models") / "best_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    print(f"Model saved successfully at: {model_path}")

    return model_path



if __name__=="__main__":
    print("In model save.py")
    path = Path("models") / "best_model.pkl"
    print("Path is: ", path)
    