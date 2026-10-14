

import joblib
from pathlib import Path

def save_object(object, object_path ):
    # model_path = Path("models") / "best_model.pkl"
    # model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(object, object_path )
    print(f"Model saved successfully at: {object_path}")

    return object_path



if __name__=="__main__":
    print("In model save.py")
    path = Path("models") / "best_model.pkl"
    print("Path is: ", path)
    