import os
import sys
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_utils import (
    CNN_IMAGE_SIZE,
    LABEL_NAMES,
    get_project_root,
    load_cnn_dataset,
    print_metrics,
)


def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    base_dir = get_project_root()
    train_dir = os.path.join(base_dir, 'dataset', 'train')
    test_dir = os.path.join(base_dir, 'dataset', 'test')
    demo_dir = os.path.join(base_dir, 'demo')

    print("=== Step 1: Load dataset ===")
    X_train, y_train = load_cnn_dataset(train_dir)
    print("---")
    X_test, y_test = load_cnn_dataset(test_dir)

    print(f"\nDataset loaded. Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    if len(X_train) == 0 or len(X_test) == 0:
        print("[ERROR] Missing training or test images. Please check the dataset structure.")
        return

    y_train_one_hot = to_categorical(y_train, num_classes=len(LABEL_NAMES))
    input_shape = (CNN_IMAGE_SIZE[1], CNN_IMAGE_SIZE[0], 3)

    print("\n=== Step 2: Train small CNN ===")
    model = build_model(input_shape, len(LABEL_NAMES))
    callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    model.fit(
        X_train,
        y_train_one_hot,
        validation_split=0.2,
        epochs=15,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    print("\n=== Step 3: Evaluate model ===")
    prediction_probs = model.predict(X_test, batch_size=32, verbose=0)
    y_pred = np.argmax(prediction_probs, axis=1)
    print_metrics(y_test, y_pred)
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES, zero_division=0))

    os.makedirs(demo_dir, exist_ok=True)
    model_path = os.path.join(demo_dir, 'rps_small_cnn.keras')
    model.save(model_path)
    print(f"[OK] Model saved to: {model_path}")


if __name__ == '__main__':
    main()
