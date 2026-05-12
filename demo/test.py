import os
import sys
import argparse
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_utils import (
    LABEL_NAMES,
    get_project_root,
    iter_labeled_image_paths,
    preprocess_cnn_image,
    preprocess_efficientnet_image,
    preprocess_flatten_image,
    preprocess_mobilenet_image,
)


def load_model(model_path):
    ext = os.path.splitext(model_path)[1].lower()
    if ext == '.pkl':
        model = joblib.load(model_path)
        if hasattr(model, 'named_steps') and 'hog' in model.named_steps:
            model_type = 'pkl_hog'
        else:
            model_type = 'pkl_flat'
        return model, model_type
    if ext in ('.h5', '.keras'):
        from tensorflow.keras.models import load_model as keras_load_model
        model = keras_load_model(model_path)
        model_name = os.path.basename(model_path).lower()
        if 'efficientnet' in model_name:
            model_type = 'keras_efficientnet'
        elif 'mobilenet' in model_name:
            model_type = 'keras_mobilenet'
        else:
            model_type = 'keras_cnn'
        return model, model_type
    raise ValueError(f'不支援的模型格式: {ext}')


def collect_predictions(model, model_type, test_dir):
    y_true = []
    y_pred = []
    keras_batch = []
    keras_labels = []

    for image_path, label_idx in iter_labeled_image_paths(test_dir):
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            continue

        if model_type == 'keras_mobilenet':
            keras_batch.append(preprocess_mobilenet_image(image))
            keras_labels.append(label_idx)
            continue

        if model_type == 'keras_efficientnet':
            keras_batch.append(preprocess_efficientnet_image(image))
            keras_labels.append(label_idx)
            continue

        if model_type == 'keras_cnn':
            keras_batch.append(preprocess_cnn_image(image))
            keras_labels.append(label_idx)
            continue

        if model_type == 'pkl_hog':
            prediction = model.predict([image])[0]
            y_true.append(label_idx)
            y_pred.append(int(prediction))
            continue

        features = preprocess_flatten_image(image).reshape(1, -1)
        prediction = model.predict(features)[0]
        y_true.append(label_idx)
        y_pred.append(int(prediction))

    if model_type in ('keras_mobilenet', 'keras_efficientnet', 'keras_cnn') and keras_batch:
        X_test = np.array(keras_batch, dtype=np.float32)
        prediction_probs = model.predict(X_test, batch_size=32, verbose=0)
        y_true = keras_labels
        y_pred = np.argmax(prediction_probs, axis=1).tolist()

    return np.array(y_true), np.array(y_pred)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model on the RPS test set')
    parser.add_argument('-m', '--model', type=str, default='rps_svm_model.pkl', help='Path to the model file')
    args = parser.parse_args()

    base_dir = get_project_root()
    default_model_path = os.path.join(base_dir, 'demo', args.model)
    model_path = args.model if os.path.isabs(args.model) else default_model_path
    test_dir = os.path.join(base_dir, 'dataset', 'test')

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return

    if not os.path.exists(test_dir):
        print(f"[ERROR] Test dataset not found: {test_dir}")
        return

    print('[LOAD] Loading model...')
    model, model_type = load_model(model_path)
    print('[OK] Model loaded.\n')

    print('[LOAD] Reading test images and running predictions...')
    y_true, y_pred = collect_predictions(model, model_type, test_dir)

    if len(y_true) == 0:
        print('[ERROR] No test images were loaded. Please check the dataset structure.')
        return

    accuracy = accuracy_score(y_true, y_pred)
    print("\nTest summary:")
    print(f"Total test images: {len(y_true)}")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    print('Classification report:')
    print(classification_report(y_true, y_pred, labels=list(range(len(LABEL_NAMES))), target_names=LABEL_NAMES, zero_division=0))


if __name__ == '__main__':
    main()
