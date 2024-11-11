import numpy as np
from typing import List, Union
import onnxruntime as ort
import torch
import os
import cv2

# Загрузка ONNX модели
onnx_model_path = "yolo11-withDiffSe.onnx"
try:
    ort_session = ort.InferenceSession(onnx_model_path)
except Exception as e:
    raise FileNotFoundError(f"ONNX model file '{onnx_model_path}' not found or failed to load. Error: {e}")

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess the image for ONNX model input."""
    # Resize and normalize the image as per model requirements
    img_resized = cv2.resize(image, (640, 640))
    img_normalized = img_resized / 255.0  # normalize to [0, 1]
    img_transposed = np.transpose(img_normalized, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    img_expanded = np.expand_dims(img_transposed, axis=0)  # add batch dimension
    return img_expanded.astype(np.float32)

def infer_image_bbox(image: np.ndarray) -> List[dict]:
    """Функция для получения ограничивающих рамок объектов на изображении.

    Args:
        image (np.ndarray): Изображение, на котором будет производиться инференс.

    Returns:
        List[dict]: Список словарей с координатами ограничивающих рамок и оценками.
    """
    res_list = []

    # Препроцессинг изображения
    input_image = preprocess_image(image)
    input_name = ort_session.get_inputs()[0].name

    # ONNX Inference
    output = ort_session.run(None, {input_name: input_image})

    # Обработка выходных данных
    for box in output[0]:  # Assuming output[0] contains bounding box data
        xc, yc, w, h, conf = box[:5]

        formatted = {
            'xc': xc,
            'yc': yc,
            'w': w,
            'h': h,
            'label': 0,  # Placeholder for label, replace if model provides
            'score': conf
        }
        res_list.append(formatted)

    return res_list

def predict(images: Union[List[np.ndarray], np.ndarray]) -> List[List[dict]]:
    """Функция производит инференс модели на одном или нескольких изображениях.

    Args:
        images (Union[List[np.ndarray], np.ndarray]): Список изображений или одно изображение.

    Returns:
        List[List[dict]]: Список списков словарей с результатами предикта
        на найденных изображениях.
    """
    results = []
    if isinstance(images, np.ndarray):
        images = [images]

    # Обрабатываем каждое изображение из полученного списка
    for image in images:
        image_results = infer_image_bbox(image)
        results.append(image_results)

    return results

# Создаем тестовое изображение (640x640, 3 канала, заполняем белым цветом)
test_image = np.full((640, 640, 3), 255, dtype=np.uint8)

# Проверяем функцию preprocess_image
preprocessed_image = preprocess_image(test_image)
print("Preprocessed image shape:", preprocessed_image.shape)
print("Preprocessed image values (min, max):", preprocessed_image.min(), preprocessed_image.max())

# Проверяем функцию infer_image_bbox
try:
    bbox_results = infer_image_bbox(test_image)
    print("Bounding box results:")
    for bbox in bbox_results:
        print(bbox)
except Exception as e:
    print("Error during inference:", e)

# Проверяем функцию predict
try:
    predictions = predict([test_image])
    print("Prediction results for multiple images:")
    for i, prediction in enumerate(predictions):
        print(f"Image {i+1} predictions:")
        for bbox in prediction:
            print(bbox)
except Exception as e:
    print("Error during prediction:", e)