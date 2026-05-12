import os
import sys
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_utils import HOGFeatureExtractor, get_project_root, load_raw_images_from_folder, print_metrics


def main():
    base_dir = get_project_root()
    train_dir = os.path.join(base_dir, 'dataset', 'train')
    test_dir = os.path.join(base_dir, 'dataset', 'test')
    demo_dir = os.path.join(base_dir, 'demo')

    print("=== Step 1: Load dataset ===")
    X_train, y_train = load_raw_images_from_folder(train_dir)
    print("---")
    X_test, y_test = load_raw_images_from_folder(test_dir)

    print(f"\nDataset loaded. Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    if len(X_train) == 0 or len(X_test) == 0:
        print("[ERROR] Missing training or test images. Please check the dataset structure.")
        return

    print("\n=== Step 2: Train HOG + SVM ===")
    pipeline = Pipeline([
        ('hog', HOGFeatureExtractor()),
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=5.0, gamma='scale')),
    ])
    pipeline.fit(X_train, y_train)

    print("\n=== Step 3: Evaluate model ===")
    y_pred = pipeline.predict(X_test)
    print_metrics(y_test, y_pred)

    os.makedirs(demo_dir, exist_ok=True)
    model_path = os.path.join(demo_dir, 'rps_hog_svm_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"[OK] Model saved to: {model_path}")


if __name__ == '__main__':
    main()
