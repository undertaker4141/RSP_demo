import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_utils import (
    preprocess_cnn_image,
    preprocess_efficientnet_image,
    preprocess_flatten_image,
    preprocess_mobilenet_image,
)


def load_smart_model(model_path):
    ext = os.path.splitext(model_path)[1].lower()

    if ext == '.pkl':
        print('[ML] Traditional model (.pkl) detected')
        import joblib
        model = joblib.load(model_path)
        if hasattr(model, 'named_steps') and 'hog' in model.named_steps:
            return model, 'pkl_hog'
        return model, 'pkl_flat'

    if ext in ['.h5', '.keras']:
        print('[TF] TensorFlow/Keras model detected')
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        model_name = os.path.basename(model_path).lower()
        if 'efficientnet' in model_name:
            model_type = 'keras_efficientnet'
        elif 'mobilenet' in model_name:
            model_type = 'keras_mobilenet'
        else:
            model_type = 'keras_cnn'
        return model, model_type

    if ext in ['.pt', '.pth']:
        print('[Torch] PyTorch model detected')
        import torch
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        return model, 'torch'

    raise ValueError(f'❌ 不支援的模型格式: {ext}')


def preprocess_and_predict(model, model_type, roi):
    if model_type == 'pkl_flat':
        features = preprocess_flatten_image(roi).reshape(1, -1)
        prediction = model.predict(features)[0]
        scores = None
        if hasattr(model, 'predict_proba'):
            scores = model.predict_proba(features)[0]
        return int(prediction), scores

    if model_type == 'pkl_hog':
        prediction = model.predict([roi])[0]
        scores = None
        if hasattr(model, 'predict_proba'):
            scores = model.predict_proba([roi])[0]
        return int(prediction), scores

    if model_type == 'keras_mobilenet':
        features = np.expand_dims(preprocess_mobilenet_image(roi), axis=0)
        prediction_probs = model.predict(features, verbose=0)[0]
        return int(np.argmax(prediction_probs)), prediction_probs

    if model_type == 'keras_efficientnet':
        features = np.expand_dims(preprocess_efficientnet_image(roi), axis=0)
        prediction_probs = model.predict(features, verbose=0)[0]
        return int(np.argmax(prediction_probs)), prediction_probs

    if model_type == 'keras_cnn':
        features = np.expand_dims(preprocess_cnn_image(roi), axis=0)
        prediction_probs = model.predict(features, verbose=0)[0]
        return int(np.argmax(prediction_probs)), prediction_probs

    if model_type == 'torch':
        import torch
        rgb = preprocess_mobilenet_image(roi)
        features = np.expand_dims(rgb.transpose((2, 0, 1)), axis=0)
        tensor_features = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(tensor_features)
            probabilities = torch.softmax(outputs, dim=1)
            scores = probabilities[0].cpu().numpy()
            _, predicted = torch.max(probabilities, 1)
        return int(predicted.item()), scores

    raise ValueError(f'❌ 未知的模型類型: {model_type}')


def draw_text_with_background(frame, text, position, font_scale, text_color, bg_color, thickness=2, padding=6):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    top_left = (x - padding, y - text_height - padding)
    bottom_right = (x + text_width + padding, y + baseline + padding)
    cv2.rectangle(frame, top_left, bottom_right, bg_color, -1)
    cv2.putText(frame, text, position, font, font_scale, text_color, thickness)


def main():
    parser = argparse.ArgumentParser(description='Rock Paper Scissors Smart Inference')
    parser.add_argument('-m', '--model', type=str, default='demo/rps_svm_model.pkl', help='Path to the model file')
    args = parser.parse_args()

    model_path = args.model

    if not os.path.exists(model_path):
        print(f"❌ 錯誤：找不到模型檔案 '{model_path}'")
        return

    print(f'⏳ 載入模型 {model_path} 中...')
    try:
        model, model_type = load_smart_model(model_path)
        print('✅ 模型載入成功！')
    except Exception as error:
        print(f'❌ 載入模型失敗：{error}')
        return

    labels = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('❌ 無法開啟攝影機')
        return

    debug_roi_dir = os.path.join('demo', 'debug_roi')
    os.makedirs(debug_roi_dir, exist_ok=True)
    snapshot_index = 0

    prev_time = 0
    prediction_interval = 3
    frame_count = 0
    result_text = 'Waiting'
    score_texts = ['Rock: N/A', 'Paper: N/A', 'Scissors: N/A']

    print("🎥 啟動智慧型攝影機... (按下 'q' 離開)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print('❌ 無法從攝影機讀取畫面')
            break

        h, w, _ = frame.shape
        box_size = 300
        start_x = max(0, w // 2 - box_size // 2)
        start_y = max(0, h // 2 - box_size // 2)
        end_x = min(w, start_x + box_size)
        end_y = min(h, start_y + box_size)
        roi = frame[start_y:end_y, start_x:end_x]
        roi_preview = roi.copy()

        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
        cv2.putText(frame, 'Put Hand Here', (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        try:
            if frame_count % prediction_interval == 0:
                prediction, scores = preprocess_and_predict(model, model_type, roi)
                result_text = labels.get(prediction, 'Unknown')
                if scores is not None and len(scores) >= 3:
                    score_texts = [
                        f'Rock: {float(scores[0]):.3f}',
                        f'Paper: {float(scores[1]):.3f}',
                        f'Scissors: {float(scores[2]):.3f}',
                    ]
                else:
                    score_texts = ['Rock: N/A', 'Paper: N/A', 'Scissors: N/A']
        except Exception as error:
            result_text = 'Error'
            print(f'⚠️ 預測時發生錯誤: {error}')
            break

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        frame_count += 1

        draw_text_with_background(frame, f'Prediction: {result_text}', (10, 40), 1.1, (0, 255, 0), (0, 0, 0), 2)
        draw_text_with_background(frame, f'FPS: {fps:.1f}', (10, 78), 0.9, (255, 0, 0), (0, 0, 0), 2)
        draw_text_with_background(frame, f'Type: {model_type.upper()}', (10, 116), 0.8, (255, 255, 0), (0, 0, 0), 2)

        panel_top = max(0, h - 120)
        cv2.rectangle(frame, (0, panel_top), (330, h), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, panel_top), (330, h), (80, 80, 80), 1)
        draw_text_with_background(frame, score_texts[0], (10, panel_top + 30), 0.75, (255, 255, 255), (0, 0, 0), 2)
        draw_text_with_background(frame, score_texts[1], (10, panel_top + 62), 0.75, (255, 255, 255), (0, 0, 0), 2)
        draw_text_with_background(frame, score_texts[2], (10, panel_top + 94), 0.75, (255, 255, 255), (0, 0, 0), 2)

        roi_preview_resized = cv2.resize(roi_preview, (256, 256))
        cv2.putText(roi_preview_resized, 'ROI Preview', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('ROI Preview', roi_preview_resized)
        cv2.imshow('Smart Camera Inference', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            snapshot_path = os.path.join(debug_roi_dir, f'roi_{snapshot_index:03d}.png')
            cv2.imwrite(snapshot_path, roi_preview)
            print(f'[SAVE] ROI snapshot saved: {snapshot_path}')
            snapshot_index += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
