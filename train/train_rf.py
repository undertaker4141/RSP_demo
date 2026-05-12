import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_utils import get_project_root, load_flatten_dataset, print_metrics


def main():
    base_dir = get_project_root()
    train_dir = os.path.join(base_dir, 'dataset', 'train')
    test_dir = os.path.join(base_dir, 'dataset', 'test')
    demo_dir = os.path.join(base_dir, 'demo')

    print("=== Step 1: Load dataset ===")
    X_train, y_train = load_flatten_dataset(train_dir)
    print("---")
    X_test, y_test = load_flatten_dataset(test_dir)

    print(f"\nDataset loaded. Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    if len(X_train) == 0 or len(X_test) == 0:
        print("[ERROR] Missing training or test images. Please check the dataset structure.")
        return

    print("\n=== Step 2: Train Random Forest ===")
    model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("\n=== Step 3: Evaluate model ===")
    y_pred = model.predict(X_test)
    print_metrics(y_test, y_pred)

    os.makedirs(demo_dir, exist_ok=True)
    model_path = os.path.join(demo_dir, 'rps_rf_model.pkl')
    joblib.dump(model, model_path)
    print(f"[OK] Model saved to: {model_path}")


if __name__ == '__main__':
    main()
