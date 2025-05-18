import time
from collections import defaultdict
from datetime import datetime
import cv2
from torch.xpu import device
from ultralytics import YOLO
import numpy as np
import importlib.util

YOLO_CONFIG = {
    "MODEL_TYPE": "yolo12n",
    'MODEL_PATH': "C:/Users/s8369084/PycharmProjects/smart-datak/smart-datak-api/algo-api/models/YOLO/model-v1.6.torchscript"
}

INFERENCE_CONFIG = {
    "DEVICE": "cuda",
    "CONF_THRESHOLD": 0.45,
    "IMG_SIZE": (1280, 736),
    "BATCH_SIZE": 16,
    "warmup_iterations_length": 20,
    "dummy_img_path": "/na-0108-c02a/public$/EFFI/פרויקטים מיוחדים/מערכות לזיהות פלטפורמות/datasets/datak_val_data"
}

PLATFORMS_LIST = ('Baz', 'Raam', 'Barak', 'Sufa', 'Adir')


class YOLODetector:
    def __init__(self, config_path="..config/yolo_config.py"):
        self.model_config = YOLO_CONFIG
        self.inference_config = INFERENCE_CONFIG

        self._set_inference_parameters()

        self.model = self._load_model()
        # self.model.to(device='cuda')

        # self._warmup()

    def _load_model(self):
        return YOLO(self.model_config["MODEL_PATH"])

    def _set_inference_parameters(self):
        self.imgsz = self.inference_config['IMG_SIZE']
        self.conf_threshold = self.inference_config['CONF_THRESHOLD']

    def _warmup(self):
        img = np.zeros((self.imgsz[1], self.imgsz[0], 3), dtype=np.uint8)
        self.iter_length = self.inference_config['warmup_iterations_length']
        for i in range(self.iter_length):
            self.model.predict(source=img, imgsz=self.imgsz, device='cuda')

    def detect_objects(self, images):
        # images = self._validate_images(images)
        # print(f"model device is: {self.model.device}")
        detections = self.model.predict(images, imgsz=self.imgsz, conf=self.conf_threshold, device='cuda')

        return detections

    def format_detections(self, results, url):
        """formats raw yolo detections to a desired dict"""
        # print("entered format")
        class_counts = self.get_class_counts(results)
        print(class_counts)

        is_platform_exists = False
        platform_type = ""

        platform_type = self.get_platform(class_counts)
        if platform_type:
            is_platform_exists = True

        person_count = self.get_person_count(class_counts)

        is_active_datak = False

        if person_count > 0 and is_platform_exists:
            is_active_datak = True

        return {
            "channel_id": url,
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "isPlatform_Exists": is_platform_exists,
            "platformType": platform_type,
            "numOfPersons": person_count,
            "isActiveDatak": is_active_datak,
        }

    def get_class_counts(self, results):
        class_counts = defaultdict(int)
        for res in results:
            for box in res.boxes:
                conf = box.conf[0].item()
                if conf >= self.conf_threshold:
                    class_id = int(box.cls[0])
                    class_name = res.names[class_id]
                    class_counts[class_name] += 1

        # formatted_counts = [{"class": cls, "count": count} for cls, count in class_counts.items()]
        return class_counts

    @staticmethod
    def get_platform(detections):
        detected_platforms = []
        for detected in detections:
            if detected in PLATFORMS_LIST:
                detected_platforms.append(detected)

        return detected_platforms

    @staticmethod
    def get_person_count(detections):
        person_count = detections.get('person', 0)
        return person_count