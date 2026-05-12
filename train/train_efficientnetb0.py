import os
import sys
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_utils import (
    EFFICIENTNET_IMAGE_SIZE,
    LABEL_NAMES,
    get_project_root,
    load_efficientnet_dataset,
    print_metrics,
)


def build_model(input_shape, num_classes):
    backbone = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    backbone.trainable = False

    model = Sequential([
        backbone,
        GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
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
    X_train, y_train = load_efficientnet_dataset(train_dir)
    print("---")
    X_test, y_test = load_efficientnet_dataset(test_dir)

    print(f"\nDataset loaded. Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    if len(X_train) == 0 or len(X_test) == 0:
        print("[ERROR] Missing training or test images. Please check the dataset structure.")
        return

    y_train_one_hot = to_categorical(y_train, num_classes=len(LABEL_NAMES))
    input_shape = (EFFICIENTNET_IMAGE_SIZE[1], EFFICIENTNET_IMAGE_SIZE[0], 3)

    print("\n=== Step 2: Train EfficientNetB0 ===")
    model = build_model(input_shape, len(LABEL_NAMES))
    callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    model.fit(
        X_train,
        y_train_one_hot,
        validation_split=0.2,
        epochs=10,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )

    print("\n=== Step 3: Evaluate model ===")
    prediction_probs = model.predict(X_test, batch_size=16, verbose=0)
    y_pred = np.argmax(prediction_probs, axis=1)
    print_metrics(y_test, y_pred)
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES, zero_division=0))

    os.makedirs(demo_dir, exist_ok=True)
    model_path = os.path.join(demo_dir, 'rps_efficientnetb0.keras')
    model.save(model_path)
    print(f"[OK] Model saved to: {model_path}")


if __name__ == '__main__':
    main()
