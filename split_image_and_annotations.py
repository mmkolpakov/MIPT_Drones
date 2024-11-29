import math
import cv2
import xml.etree.ElementTree as ET
import logging
from typing import Tuple, List, Optional, Dict, Union, Set
from pathlib import Path
import sys
import numpy as np
import csv
from collections import defaultdict
import re
from enum import Enum
import hashlib

# --------------------------- Константы ---------------------------

# Путь к директории с исходными данными (изображениями и аннотациями)
INPUT_DIR = Path(r"C:\Users\mkolp\OneDrive\Изображения\break\break")

# Путь к директории для сохранения результатов
OUTPUT_DIR = Path(r"C:\Users\mkolp\OneDrive\Изображения\break\results")

# Параметры разделения изображения
SPLIT_HORIZONTAL = 2  # Количество разбиений по горизонтали
SPLIT_VERTICAL = 2  # Количество разбиений по вертикали
OVERLAP = 0.46  # перекрытие двух соседних частей [долей от исходного]

# Форматы сохраняемых изображений
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Форматы аннотаций
SUPPORTED_ANNOTATION_FORMATS = ['.xml', '.txt']  # XML для PASCAL VOC, TXT для YOLO

# Формат сохраняемых изображений (должен соответствовать одному из SUPPORTED_IMAGE_FORMATS)
IMAGE_FORMAT = '.jpg'

# Минимальная площадь пересечения объекта с участком (для фильтрации мелких объектов)
MIN_INTERSECTION_AREA = 10  # пикселей

# Настройки логирования
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = '%(levelname)s: %(message)s'

# Путь к файлу лога CSV (будет сохранен в корне OUTPUT_DIR)
LOG_CSV_PATH = OUTPUT_DIR / "process_log.csv"

# Путь к файлу для сравнения аннотаций
DELTA_PIXELS_CSV = OUTPUT_DIR / "delta_pixels.csv"

class NamingMethod(Enum):
    COORDINATES = 1
    NUMERICAL = 2

# Способ именования: NamingMethod.COORDINATES или NamingMethod.NUMERICAL
NAMING_METHOD = NamingMethod.NUMERICAL

# Имя подпапки для дубликатов
DUPLICATES_DIR_NAME = "duplicates"

# Флаг для включения визуализации разбиений и аннотаций
VISUALIZE_SPLITS = True  # Установите True для включения визуализации

# Путь к директории для сохранения визуализаций
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

# Создание отдельных папок для каждого типа визуализаций
ORIGINAL_ANNOTATIONS_DIR = VISUALIZATION_DIR / "original_annotations"
SPLITS_DIR = VISUALIZATION_DIR / "splits"
ANNOTATED_SPLITS_DIR = VISUALIZATION_DIR / "annotated_splits"

# Флаг для совмещения разрезов и аннотаций на одном изображении
VISUALIZE_SPLITS_ON_ANNOTATIONS = True  # Установите True, чтобы совмещать разрезы и аннотации

REMOVE_OBJECT_ID = True  # Установите True, если нужно удалить поле 'object_id' из аннотаций

# -----------------------------------------------------------------

class ImageData:
    """
    Класс для хранения данных изображения.
    """

    def __init__(self, data: np.ndarray, image_size_x: int, image_size_y: int):
        self.data = data
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y

class AnnotatedImageData(ImageData):
    """
    Класс для хранения данных изображения и его аннотаций.
    """

    def __init__(self, data: np.ndarray, image_size_x: int, image_size_y: int,
                 annotation: Union[ET.Element, List[str]],
                 annotation_format: str,
                 image_path: Optional[Path] = None,
                 object_ids: Optional[List[int]] = None):
        super().__init__(data, image_size_x, image_size_y)
        self.annotation = annotation
        self.annotation_format = annotation_format  # '.xml' или '.txt'
        self.image_path = image_path  # Путь к текущему изображению
        self.object_ids = object_ids if object_ids is not None else []  # Список object_id для каждого объекта

def load_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Загружает изображение с поддержкой путей с нелатинскими символами.

    :param image_path: Путь к изображению.
    :return: Изображение в формате NumPy массива или None, если загрузка не удалась.
    """
    try:
        with image_path.open('rb') as f:
            data = f.read()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        logging.error(f"Ошибка при загрузке изображения '{image_path}': {e}")
        return None

def save_image(img: np.ndarray, output_path: Path, image_format: str) -> None:
    """
    Сохраняет изображение с поддержкой путей с нелатинскими символами.

    :param img: Изображение в формате NumPy массива.
    :param output_path: Путь для сохранения изображения.
    :param image_format: Формат изображения (например, '.jpg').
    :raises IOError: Если сохранение изображения не удалось.
    """
    try:
        # Убираем точку из формата
        format_without_dot = image_format.lstrip('.')
        success, encoded_image = cv2.imencode(f'.{format_without_dot}', img)
        if success:
            with output_path.open('wb') as f:
                f.write(encoded_image)
            logging.debug(f"Изображение сохранено: {output_path}")
        else:
            raise IOError(f"Не удалось сохранить изображение '{output_path}'")
    except Exception as e:
        logging.error(f"Ошибка при сохранении изображения '{output_path}': {e}")
        raise

def calculate_split_coordinates(
        width: int,
        height: int,
        split_h: int,
        split_v: int,
        overlap: float
) -> List[Tuple[int, int, int, int]]:
    """
    Вычисляет координаты для разбиения изображения на сетку, гарантируя покрытие всего изображения.

    :param width: Ширина изображения.
    :param height: Высота изображения.
    :param split_h: Количество разбиений по горизонтали.
    :param split_v: Количество разбиений по вертикали.
    :param overlap: Размер перекрытия соседних разрезов в долях от исходного изображения (от 0 до 1).
    :return: Список кортежей с координатами (x1, y1, x2, y2) для каждого участка.
    :raises ValueError: Если overlap не в диапазоне [0, 1), split_h или split_v <= 0, или разрезы имеют разные размеры.
    """
    if not (0 <= overlap < 1):
        raise ValueError("Overlap must be between 0 and 1.")
    if split_h <= 0 or split_v <= 0:
        raise ValueError("split_h and split_v must be greater than 0.")

    # Вычисление перекрытия в пикселях
    overlap_pixels_x = overlap * width
    overlap_pixels_y = overlap * height

    # Вычисление размеров подизображений
    sub_width = math.floor((width + (split_h - 1) * overlap_pixels_x) / split_h)
    sub_height = math.floor((height + (split_v - 1) * overlap_pixels_y) / split_v)

    coords = []

    for i in range(split_v):
        for j in range(split_h):
            x1 = int(math.floor(j * (sub_width - overlap_pixels_x)))
            y1 = int(math.floor(i * (sub_height - overlap_pixels_y)))

            #  Использование ceil для последних подкадров для гарантии покрытия
            x2 = int(math.ceil(x1 + sub_width) if j == split_h - 1 else math.floor(x1 + sub_width))
            y2 = int(math.ceil(y1 + sub_height) if i == split_v - 1 else math.floor(y1 + sub_height))

            # Корректировка, чтобы не выйти за пределы изображения (для надежности)
            x2 = min(x2, width)
            y2 = min(y2, height)

            coords.append((x1, y1, x2, y2))

    # Проверка размеров подкадров
    first_sub_width = coords[0][2] - coords[0][0]
    first_sub_height = coords[0][3] - coords[0][1]

    # Допускаем разницу в 1 пиксель для последнего ряда/столбца
    for coord in coords:
        if not ((coord[2] - coord[0] == first_sub_width or abs(coord[2] - coord[0] - first_sub_width) <= 1) and (
                coord[3] - coord[1] == first_sub_height or abs(coord[3] - coord[1] - first_sub_height) <= 1)):
            raise ValueError(f"Разрезы имеют разные размеры {coord}, проверьте параметры разреза и перекрытия.")

    # Проверка покрытия всего изображения
    covered_area = 0
    for x1, y1, x2, y2 in coords:
        covered_area += (x2 - x1) * (y2 - y1)

    if covered_area < width * height:
        raise ValueError("Разбиение не покрывает все изображение.")

    return coords

def adjust_bndbox_voc(obj: ET.Element, bndbox: ET.Element, split_coords: Tuple[int, int, int, int],
                      cropped_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int, int]]:
    """
    Корректирует координаты ограничивающего прямоугольника объекта относительно вырезанного участка для PASCAL VOC (XML).

    :param obj: Элемент объекта.
    :param bndbox: Элемент XML с координатами объекта.
    :param split_coords: Кортеж с координатами участка (x1, y1, x2, y2).
    :param cropped_size: Размеры вырезанного изображения (width, height).
    :return: Кортеж с новыми координатами (object_id, new_xmin, new_ymin, new_xmax, new_ymax) или None, если объект не попадает в разрез или имеет некорректные координаты.
    """
    x1, y1, x2, y2 = split_coords
    cropped_width, cropped_height = cropped_size

    try:
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))
    except (AttributeError, ValueError, TypeError) as e:
        logging.error(f"Некорректные координаты bndbox в объекте: {obj}: {e}")
        return None

    # Проверка пересечения с учетом минимальной площади
    intersection_area = max(0, min(xmax, x2) - max(xmin, x1)) * max(0, min(ymax, y2) - max(ymin, y1))
    if intersection_area < MIN_INTERSECTION_AREA:
        logging.debug(f"Объект {obj} не пересекается с участком или площадь пересечения меньше минимальной.")
        return None

    # Корректировка координат
    new_xmin = max(0, xmin - x1)
    new_ymin = max(0, ymin - y1)
    new_xmax = min(cropped_width, xmax - x1)
    new_ymax = min(cropped_height, ymax - y1)

    # Проверка на корректность после обрезки.
    if new_xmax <= new_xmin or new_ymax <= new_ymin:
        logging.debug(
            f"Объект {obj} обрезан некорректно: new_xmin={new_xmin}, new_ymin={new_ymin}, new_xmax={new_xmax}, new_ymax={new_ymax}")
        return None

    object_id = int(obj.find("object_id").text) if obj.find(
        "object_id") is not None else -1  # Если object_id не найден, возвращаем -1. Это должно быть обработано далее.

    return object_id, new_xmin, new_ymin, new_xmax, new_ymax

def generate_unique_object_id(image_path: str, line: str) -> int:
    """
    Генерирует уникальный object_id на основе имени файла и строки аннотации.
    """
    hash_string = f"{image_path}_{line}".encode('utf-8')
    return int(hashlib.md5(hash_string).hexdigest(), 16) % 1000000  # Ограничиваем object_id до 6 знаков

def adjust_bndbox_yolo(line: str, image_path: Path, split_coords: Tuple[int, int, int, int],
                       original_size: Tuple[int, int],
                       cropped_size: Tuple[int, int]) -> Optional[Tuple[str, int]]:
    """
    Корректирует координаты ограничивающего прямоугольника объекта относительно вырезанного участка для YOLO (TXT).

    :param line: Строка с координатами объекта в формате YOLO (class x_center y_center width height [object_id]).
    :param image_path: Путь к исходному изображению для генерации уникального object_id.
    :param split_coords: Кортеж с координатами участка (x1, y1, x2, y2).
    :param original_size: Размеры исходного изображения (width, height).
    :param cropped_size: Размеры вырезанного изображения (width, height).
    :return: Кортеж из строки с новыми координатами объекта и его object_id или None, если объект не попадает в разрез или имеет некорректные координаты.
    """
    try:
        parts = line.strip().split()

        # Проверяем наличие object_id в строке аннотации, если есть - сохраняем, если нет, генерируем.
        if len(parts) == 6:
            class_id, x_center, y_center, width, height, object_id = parts
            object_id = int(object_id)
        elif len(parts) == 5:
            class_id, x_center, y_center, width, height = parts
            object_id = generate_unique_object_id(str(image_path), line)
        else:
            raise ValueError(f"Некорректный формат строки YOLO: {line}")

        class_id = int(class_id)
        x_center, y_center, width, height = map(float, [x_center, y_center, width, height])

    except (ValueError, TypeError) as e:
        logging.error(f"Некорректные координаты bndbox YOLO в строке '{line}': {e}")
        return None

    original_width, original_height = original_size
    split_x1, split_y1, split_x2, split_y2 = split_coords

    # Преобразуем нормализованные координаты в абсолютные относительно исходного изображения
    x_center_abs = x_center * original_width
    y_center_abs = y_center * original_height
    width_abs = width * original_width
    height_abs = height * original_height

    xmin = x_center_abs - width_abs / 2
    ymin = y_center_abs - height_abs / 2
    xmax = x_center_abs + width_abs / 2
    ymax = y_center_abs + height_abs / 2

    # Проверка пересечения с разрезом
    if xmax <= split_x1 or xmin >= split_x2 or ymax <= split_y1 or ymin >= split_y2:
        return None  # Объект не пересекается

    # Корректировка координат относительно вырезанного участка
    new_xmin = max(xmin, split_x1) - split_x1
    new_ymin = max(ymin, split_y1) - split_y1
    new_xmax = min(xmax, split_x2) - split_x1
    new_ymax = min(ymax, split_y2) - split_y1

    # Проверка минимальной площади пересечения
    if (new_xmax - new_xmin) * (new_ymax - new_ymin) < MIN_INTERSECTION_AREA:
        return None

    if new_xmax <= new_xmin or new_ymax <= new_ymin:
        return None

    # Преобразуем обратно в YOLO формат относительно вырезанного изображения
    new_width = new_xmax - new_xmin
    new_height = new_ymax - new_ymin
    new_x_center = new_xmin + new_width / 2
    new_y_center = new_ymin + new_height / 2

    # Нормализуем координаты
    norm_x_center = new_x_center / cropped_size[0]
    norm_y_center = new_y_center / cropped_size[1]
    norm_width = new_width / cropped_size[0]
    norm_height = new_height / cropped_size[1]

    new_line = f"{class_id} {norm_x_center:.6f} {norm_y_center:.6f} {norm_width:.6f} {norm_height:.6f} {object_id}"

    return new_line, object_id

def get_color(idx):
    """
    Возвращает цвет в формате (B, G, R) на основе индекса.
    """
    idx = idx * 3  # Чтобы избежать похожих цветов для последовательных индексов
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def visualize_annotations(image: np.ndarray, annotations: Union[ET.Element, List[str]], annotation_format: str,
                          output_path: Path) -> None:
    """
    Визуализирует аннотации на изображении и сохраняет результат.

    :param image: Изображение в формате NumPy массива.
    :param annotations: Аннотации в формате XML или список строк YOLO.
    :param annotation_format: Формат аннотаций '.xml' или '.txt'.
    :param output_path: Путь для сохранения визуализированного изображения.
    """
    image_copy = image.copy()
    height, width = image_copy.shape[:2]

    if annotation_format == '.xml':
        for obj in annotations.findall("object"):
            bndbox = obj.find("bndbox")
            try:
                xmin = int(float(bndbox.find("xmin").text))
                ymin = int(float(bndbox.find("ymin").text))
                xmax = int(float(bndbox.find("xmax").text))
                ymax = int(float(bndbox.find("ymax").text))
                class_name = obj.find("name").text
                object_id_elem = obj.find("object_id")
                if object_id_elem is not None:
                    object_id = int(object_id_elem.text)
                else:
                    object_id = -1
                label = f"{class_name}:{object_id}"
                color = get_color(object_id)
                cv2.rectangle(image_copy, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(image_copy, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                logging.error(f"Ошибка при визуализации аннотаций XML: {e}")
    elif annotation_format == '.txt':
        for line in annotations:
            parts = line.strip().split()
            if len(parts) not in [5, 6]:
                continue
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                if len(parts) == 6:
                    object_id = int(parts[5])
                else:
                    object_id = -1
                x_center_abs = x_center * width
                y_center_abs = y_center * height
                w_abs = w * width
                h_abs = h * height
                xmin = int(x_center_abs - w_abs / 2)
                ymin = int(y_center_abs - h_abs / 2)
                xmax = int(x_center_abs + w_abs / 2)
                ymax = int(y_center_abs + h_abs / 2)
                label = f"{class_id}:{object_id}"
                color = get_color(object_id)
                cv2.rectangle(image_copy, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(image_copy, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                logging.error(f"Ошибка при визуализации аннотаций YOLO: {e}")
    else:
        logging.error(f"Неизвестный формат аннотаций для визуализации: {annotation_format}")
        return

    try:
        save_image(image_copy, output_path, IMAGE_FORMAT)
        logging.debug(f"Визуализация аннотаций сохранена: {output_path}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении визуализации аннотаций: {e}")

def visualize_splits(image: np.ndarray, split_coords_list: List[Tuple[int, int, int, int]], output_path: Path,
                     annotations: Optional[Union[ET.Element, List[str]]] = None,
                     annotation_format: Optional[str] = None) -> None:
    """
    Визуализирует разрезы (и аннотации при необходимости) на изображении и сохраняет результат.

    :param image: Изображение в формате NumPy массива.
    :param split_coords_list: Список координат разрезов.
    :param output_path: Путь для сохранения визуализированного изображения.
    :param annotations: (Необязательно) Аннотации для визуализации.
    :param annotation_format: (Необязательно) Формат аннотаций '.xml' или '.txt'.
    """
    image_copy = image.copy()

    # Отображение аннотаций, если они предоставлены
    if annotations and annotation_format:
        visualize_annotations(image_copy, annotations, annotation_format, output_path)

    for idx, (x1, y1, x2, y2) in enumerate(split_coords_list):
        color = (255, 0, 0)
        thickness = 2
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image_copy, f"Split {idx+1}", (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    try:
        save_image(image_copy, output_path, IMAGE_FORMAT)
        logging.debug(f"Визуализация разрезов сохранена: {output_path}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении визуализации разрезов: {e}")

def cut_with_annotation(annotated_image: AnnotatedImageData, cut_set: Tuple[int, int, int, int],
                        split_idx: int) -> Tuple[AnnotatedImageData, Path, int, int, Set[int]]:
    """
    Вырезает часть изображения и соответствующие аннотации.

    :param annotated_image: Объект AnnotatedImageData с исходными данными.
    :param cut_set: Кортеж с координатами разреза (x1, y1, x2, y2).
    :param split_idx: Индекс текущей части для имени файла.
    :return: Кортеж из нового AnnotatedImageData с вырезанным изображением и обновленными аннотациями,
             пути к новому изображению, позиций левого верхнего угла (x1, y1) и множества object_id объектов на подкадре.
    """
    x1, y1, x2, y2 = cut_set  # Извлечение координат

    # Вырезаем изображение
    cropped_image = annotated_image.data[y1:y2, x1:x2]
    cropped_height, cropped_width = cropped_image.shape[:2]

    # Генерация нового имени файла на основе выбранного способа именования
    base_filename = Path(annotated_image.image_path).stem if annotated_image.image_path else 'image'
    # Удаляем любой существующий суффикс '_part_*'
    base_filename = re.sub(r'_part_\d+_\d+$', '', base_filename)

    if NAMING_METHOD == NamingMethod.COORDINATES:
        # Способ именования с использованием координат
        new_filename = f"{base_filename}_part_{x1}_{y1}{IMAGE_FORMAT}"
    elif NAMING_METHOD == NamingMethod.NUMERICAL:
        # Способ именования с использованием простой нумерации
        new_filename = f"{base_filename}_{split_idx}{IMAGE_FORMAT}"
    else:
        logging.error(f"Неизвестный способ именования: {NAMING_METHOD}")
        raise ValueError(f"Неизвестный способ именования: {NAMING_METHOD}")

    # Сохраняем в той же директории, что и исходная картинка
    new_image_path = annotated_image.image_path.parent / new_filename if annotated_image.image_path else Path(
        new_filename)

    # Корректируем аннотации
    object_ids_in_crop = set()

    if annotated_image.annotation_format == '.xml':
        new_annotation_root = ET.Element("annotation")

        # Копирование всех необходимых элементов из исходной аннотации
        for child in annotated_image.annotation:
            if child.tag == "filename":
                ET.SubElement(new_annotation_root, "filename").text = new_filename
            elif child.tag == "size":
                size = ET.SubElement(new_annotation_root, "size")
                ET.SubElement(size, "width").text = str(cropped_width)
                ET.SubElement(size, "height").text = str(cropped_height)
                # Определяем глубину на основе канальности изображения
                depth = cropped_image.shape[2] if len(cropped_image.shape) == 3 else 1
                ET.SubElement(size, "depth").text = str(depth)
            elif child.tag == "object":
                # Обработка объектов будет происходить позже
                continue
            else:
                # Копирование других элементов без изменений
                new_annotation_root.append(child)

        # Обработка объектов
        for obj in annotated_image.annotation.findall("object"):
            bndbox = obj.find("bndbox")
            adjusted_result = adjust_bndbox_voc(obj, bndbox, cut_set, (cropped_width, cropped_height))
            if adjusted_result is None:
                continue  # Объект не пересекается или пересечение слишком мало

            object_id, new_xmin, new_ymin, new_xmax, new_ymax = adjusted_result

            new_obj = ET.SubElement(new_annotation_root, "object")
            # Копирование всех подэлементов объекта кроме bndbox
            for elem in obj:
                if elem.tag != "bndbox":
                    new_elem = ET.SubElement(new_obj, elem.tag)
                    new_elem.text = elem.text

            # Создание нового bndbox
            new_bndbox = ET.SubElement(new_obj, "bndbox")
            ET.SubElement(new_bndbox, "xmin").text = str(new_xmin)
            ET.SubElement(new_bndbox, "ymin").text = str(new_ymin)
            ET.SubElement(new_bndbox, "xmax").text = str(new_xmax)
            ET.SubElement(new_bndbox, "ymax").text = str(new_ymax)

            object_ids_in_crop.add(object_id)

        new_annotation = new_annotation_root

    elif annotated_image.annotation_format == '.txt':

        new_yolo_annotations = []

        original_size = (annotated_image.image_size_x, annotated_image.image_size_y)

        for idx, line in enumerate(annotated_image.annotation):

            # Передаем image_path в adjust_bndbox_yolo

            adjusted_result = adjust_bndbox_yolo(line, annotated_image.image_path, cut_set, original_size,
                                                 (cropped_width, cropped_height))

            if adjusted_result is not None:
                adjusted_ann, object_id = adjusted_result

                new_yolo_annotations.append(adjusted_ann)

                object_ids_in_crop.add(object_id)

        new_annotation = new_yolo_annotations

    else:
        logging.error(f"Неизвестный формат аннотаций: {annotated_image.annotation_format}")
        new_annotation = None

    split_image_data = AnnotatedImageData(
        data=cropped_image,
        image_size_x=cropped_width,
        image_size_y=cropped_height,
        annotation=new_annotation,
        annotation_format=annotated_image.annotation_format,
        image_path=new_image_path,  # Устанавливаем путь к новому изображению
        object_ids=list(object_ids_in_crop)
    )

    return split_image_data, new_image_path, x1, y1, object_ids_in_crop

def verify(annotated_image: AnnotatedImageData, cut_set: Tuple[int, int, int, int]) -> Tuple[
    int, float, Tuple[bool, bool]]:
    """
    Проверяет, сколько объектов пересекают границы разреза и вычисляет минимальную долю объекта внутри.

    :param annotated_image: Объект AnnotatedImageData с исходными данными.
    :param cut_set: Кортеж с координатами разреза (x1, y1, x2, y2).
    :return: Кортеж из количества пересечённых объектов, минимальной доли объекта внутри и осей пересечения.
    """
    x1, y1, x2, y2 = cut_set
    N = 0
    min_percent = 1.0
    axis = (False, False)  # (x_axis, y_axis)

    if annotated_image.annotation_format == '.xml':
        for obj in annotated_image.annotation.findall("object"):
            bndbox = obj.find("bndbox")
            try:
                obj_xmin = int(bndbox.find("xmin").text)
                obj_ymin = int(bndbox.find("ymin").text)
                obj_xmax = int(bndbox.find("xmax").text)
                obj_ymax = int(bndbox.find("ymax").text)
            except (AttributeError, ValueError):
                continue

            # Проверка пересечения
            if obj_xmin < x2 and obj_xmax > x1 and obj_ymin < y2 and obj_ymax > y1:
                # Проверяем, полностью ли объект внутри разреза
                if obj_xmin >= x1 and obj_xmax <= x2 and obj_ymin >= y1 and obj_ymax <= y2:
                    # Объект полностью внутри разреза, не считаем его как пересекающий
                    intersect_area = (obj_xmax - obj_xmin) * (obj_ymax - obj_ymin)
                    percent_inside = intersect_area / ((obj_xmax - obj_xmin) * (obj_ymax - obj_ymin))
                    min_percent = min(min_percent, percent_inside)
                    continue
                else:
                    # Объект пересекает границы разреза
                    N += 1
                    # Вычисление доли объекта внутри разреза
                    intersect_xmin = max(obj_xmin, x1)
                    intersect_ymin = max(obj_ymin, y1)
                    intersect_xmax = min(obj_xmax, x2)
                    intersect_ymax = min(obj_ymax, y2)
                    intersect_area = max(intersect_xmax - intersect_xmin, 0) * max(intersect_ymax - intersect_ymin, 0)
                    obj_area = (obj_xmax - obj_xmin) * (obj_ymax - obj_ymin)
                    if obj_area > 0:
                        percent_inside = intersect_area / obj_area
                        min_percent = min(min_percent, percent_inside)
                        # Определение осей пересечения
                        if obj_xmin < x1 or obj_xmax > x2:
                            axis = (True, axis[1])
                        if obj_ymin < y1 or obj_ymax > y2:
                            axis = (axis[0], True)
        return N, min_percent, axis

    elif annotated_image.annotation_format == '.txt':
        for line in annotated_image.annotation:
            parts = line.strip().split()
            if len(parts) not in [5, 6]:
                continue
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                continue
            img_width, img_height = annotated_image.image_size_x, annotated_image.image_size_y
            obj_xmin = (x_center - width / 2) * img_width
            obj_ymin = (y_center - height / 2) * img_height
            obj_xmax = (x_center + width / 2) * img_width
            obj_ymax = (y_center + height / 2) * img_height

            # Проверка пересечения
            if obj_xmin < x2 and obj_xmax > x1 and obj_ymin < y2 and obj_ymax > y1:
                # Проверяем, полностью ли объект внутри разреза
                if obj_xmin >= x1 and obj_xmax <= x2 and obj_ymin >= y1 and obj_ymax <= y2:
                    # Объект полностью внутри разреза, не считаем его как пересекающий
                    intersect_area = (obj_xmax - obj_xmin) * (obj_ymax - obj_ymin)
                    percent_inside = intersect_area / ((obj_xmax - obj_xmin) * (obj_ymax - obj_ymin))
                    min_percent = min(min_percent, percent_inside)
                    continue
                else:
                    # Объект пересекает границы разреза
                    N += 1
                    # Вычисление доли объекта внутри разреза
                    intersect_xmin = max(obj_xmin, x1)
                    intersect_ymin = max(obj_ymin, y1)
                    intersect_xmax = min(obj_xmax, x2)
                    intersect_ymax = min(obj_ymax, y2)
                    intersect_area = max(intersect_xmax - intersect_xmin, 0) * max(intersect_ymax - intersect_ymin, 0)
                    obj_area = (obj_xmax - obj_xmin) * (obj_ymax - obj_ymin)
                    if obj_area > 0:
                        percent_inside = intersect_area / obj_area
                        min_percent = min(min_percent, percent_inside)
                        # Определение осей пересечения
                        if obj_xmin < x1 or obj_xmax > x2:
                            axis = (True, axis[1])
                        if obj_ymin < y1 or obj_ymax > y2:
                            axis = (axis[0], True)
        return N, min_percent, axis

    else:
        logging.error(f"Неизвестный формат аннотаций: {annotated_image.annotation_format}")
        return N, min_percent, axis

def multi_cut(annotated_image: AnnotatedImageData) -> List[Tuple[AnnotatedImageData, Path, int, int, Set[int], Tuple[int, int, int, int]]]:
    """
    Разделяет изображение и аннотации на сетку с заданными параметрами, корректируя разрезы для сохранения размеров и позиций.

    :param annotated_image: Объект AnnotatedImageData с исходными данными.
    :return: Список кортежей из AnnotatedImageData с вырезанными частями, пути к новым изображениям, позиций (x1, y1), множеств object_id и координат разрезов.
    """
    # Вычисление базовых координат разреза
    split_coords_list = calculate_split_coordinates(
        annotated_image.image_size_x,
        annotated_image.image_size_y,
        SPLIT_HORIZONTAL,
        SPLIT_VERTICAL,
        OVERLAP  # перекрытие
    )
    total_parts = len(split_coords_list)
    logging.info(f"Разделение изображения на {total_parts} частей.")

    # Параметры сдвига
    step = 15  # шаг смещения в пикселях
    nstep = 10  # количество попыток сдвига

    image_result = []  # список вырезанных изображений и их позиций

    for idx, base_split in enumerate(split_coords_list, start=1):
        logging.info(f"Обработка части {idx} из {total_parts} (Координаты: {base_split})...")
        a_base, b_base, c_base, d_base = base_split

        best_cut = base_split
        best_min_percent = 0.0
        best_N = float('inf')
        best_axis = (True, True)  # По умолчанию смещаем по обеим осям
        done = False

        for attempt in range(nstep):
            if attempt == 0:
                da, dc = 0, 0
            else:
                # Используем best_axis для определения направлений смещения
                da = -step * attempt if best_axis[0] else 0
                dc = -step * attempt if best_axis[1] else 0

            # Смещение разреза по необходимым осям
            current_split = (
                a_base + da,
                b_base + dc,
                c_base + da,
                d_base + dc
            )

            # Вычисляем размеры разреза
            split_width = current_split[2] - current_split[0]
            split_height = current_split[3] - current_split[1]

            # Корректируем координаты, чтобы разрез полностью помещался в изображение
            if current_split[0] < 0:
                current_split = (
                    0,
                    current_split[1],
                    split_width,
                    current_split[3]
                )
            if current_split[2] > annotated_image.image_size_x:
                current_split = (
                    annotated_image.image_size_x - split_width,
                    current_split[1],
                    annotated_image.image_size_x,
                    current_split[3]
                )
            if current_split[1] < 0:
                current_split = (
                    current_split[0],
                    0,
                    current_split[2],
                    split_height
                )
            if current_split[3] > annotated_image.image_size_y:
                current_split = (
                    current_split[0],
                    annotated_image.image_size_y - split_height,
                    current_split[2],
                    annotated_image.image_size_y
                )

            # Проверяем, что разрез корректен
            if current_split[0] < 0 or current_split[1] < 0 or current_split[2] > annotated_image.image_size_x or \
                    current_split[3] > annotated_image.image_size_y:
                logging.debug(f"Попытка {attempt + 1}: разрез выходит за пределы изображения {current_split}. Пропуск.")
                continue

            if current_split[0] >= current_split[2] or current_split[1] >= current_split[3]:
                logging.debug(f"Попытка {attempt + 1}: некорректный разрез {current_split}. Пропуск.")
                continue

            # Проверяем, не пересекает ли текущий разрез объекты
            N, min_percent, axis = verify(annotated_image, current_split)
            logging.debug(f"Попытка {attempt + 1}: N={N}, min_percent={min_percent:.2f}, axis={axis}")

            if N == 0:
                best_cut = current_split
                done = True
                logging.debug("Идеальный разрез найден.")
                break
            else:
                if min_percent > best_min_percent or (min_percent == best_min_percent and N < best_N):
                    best_min_percent = min_percent
                    best_cut = current_split
                    best_N = N
                    best_axis = axis

        if not done:
            logging.info(f"Идеальный разрез не найден для части {idx}. Выбирается наилучший доступный вариант.")

        # Вырезаем изображение и аннотации по выбранному разрезу
        split_image_data, split_image_path, split_x, split_y, object_ids_in_crop = cut_with_annotation(annotated_image,
                                                                                                       best_cut, idx)
        image_result.append((split_image_data, split_image_path, split_x, split_y, object_ids_in_crop, best_cut))

    return image_result

global_object_id_counter = 1

def process_all_images(input_dir: Path, output_dir: Path) -> None:
    """
    Обрабатывает все изображения и аннотации в указанной директории и её подпапках.

    :param input_dir: Путь к директории с исходными изображениями и аннотациями.
    :param output_dir: Путь к директории для сохранения разделенных изображений и аннотаций.
    """
    global global_object_id_counter

    if not input_dir.is_dir():
        logging.error(f"Входная директория не существует или не является директорией: {input_dir}")
        sys.exit(1)

    # # Поиск всех изображений в директории и подпапках
    # image_files = [p for p in input_dir.rglob('*') if p.suffix.lower() in SUPPORTED_IMAGE_FORMATS]

    # Поиск всех изображений в директории и подпапках, исключая OUTPUT_DIR
    image_files = [p for p in input_dir.rglob('*') if
                   p.suffix.lower() in SUPPORTED_IMAGE_FORMATS and not p.resolve().is_relative_to(output_dir.resolve())]

    if not image_files:
        logging.warning(f"Входная директория '{input_dir}' не содержит поддерживаемых изображений.")
        return

    logging.info(f"Найдено {len(image_files)} изображений для обработки.")

    # Создание директории для дубликатов
    duplicates_dir = output_dir / DUPLICATES_DIR_NAME
    duplicates_dir.mkdir(parents=True, exist_ok=True)

    # Создание директории для визуализаций
    if VISUALIZE_SPLITS:
        ORIGINAL_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        ANNOTATED_SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # Инициализация CSV лога
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with LOG_CSV_PATH.open('w', newline='', encoding='utf-8') as csvfile:
            log_writer = csv.writer(csvfile)
            # Запись заголовков
            log_writer.writerow([
                'Original Image',
                'Original Annotation',
                'Split Image',
                'Split Position X',
                'Split Position Y',
                'Split Annotation'
            ])

            for image_path in image_files:
                # Поиск соответствующего файла аннотаций
                annotation_path = None
                for ext in SUPPORTED_ANNOTATION_FORMATS:
                    potential_annotation = image_path.with_suffix(ext)
                    if potential_annotation.is_file():
                        annotation_path = potential_annotation
                        break

                if not annotation_path:
                    logging.warning(f"Для изображения '{image_path}' не найден соответствующий файл аннотаций.")
                    continue

                try:
                    relative_path = image_path.relative_to(input_dir).parent
                except ValueError:
                    logging.error(
                        f"Изображение '{image_path}' не находится внутри входной директории '{input_dir}'. Пропуск.")
                    continue

                # Создание соответствующей директории в OUTPUT_DIR
                image_output_dir = output_dir / relative_path
                image_output_dir.mkdir(parents=True, exist_ok=True)

                # Парсинг аннотаций
                annotation_format = annotation_path.suffix.lower()
                object_ids = []  # Список object_id для объектов

                if annotation_format == '.xml':
                    logging.info(f"Парсинг XML-аннотаций из '{annotation_path}'...")
                    try:
                        tree = ET.parse(str(annotation_path))
                        root = tree.getroot()
                        annotation = root

                        # Используем глобальный счетчик object_id для XML, проверяем существование object_id
                        for obj in annotation.findall("object"):
                            if obj.find("object_id") is None:
                                object_id_elem = ET.Element("object_id")
                                object_id_elem.text = str(global_object_id_counter)
                                obj.append(object_id_elem)
                                object_ids.append(global_object_id_counter)
                                global_object_id_counter += 1
                            else:
                                # object_id уже существует, добавляем в список object_ids
                                object_ids.append(int(obj.find("object_id").text))

                    except ET.ParseError as e:
                        logging.error(f"Ошибка парсинга XML-аннотаций: {e}")
                        continue
                elif annotation_format == '.txt':
                    logging.info(f"Парсинг YOLO-аннотаций из '{annotation_path}'...")
                    try:
                        yolo_annotations = [line.strip() for line in
                                            annotation_path.read_text(encoding='utf-8').splitlines() if line.strip()]
                        annotation = yolo_annotations  # Список строк для корректной передачи в AnnotatedImageData

                        # Нумерация объектов
                        for line in yolo_annotations:
                            parts = line.strip().split()
                            if len(parts) == 6:
                                object_id = int(parts[5])
                                object_ids.append(object_id)
                            elif len(parts) == 5:
                                object_id = generate_unique_object_id(str(image_path), line)
                                object_ids.append(object_id)
                            else:
                                logging.warning(f"Некорректная строка аннотации YOLO: '{line}'")
                                continue

                    except Exception as e:
                        logging.error(f"Ошибка чтения YOLO-аннотаций: {e}")
                        continue
                else:
                    logging.error(f"Формат аннотаций '{annotation_format}' не поддерживается.")
                    continue

                # Загрузка изображения с поддержкой нелатинских символов
                image = load_image(image_path)
                if image is None:
                    logging.error(f"Изображение не найдено или не может быть загружено: {image_path}")
                    continue

                height, width = image.shape[:2]
                depth = image.shape[2] if len(image.shape) == 3 else 1

                annotated_image = AnnotatedImageData(
                    data=image,
                    image_size_x=width,
                    image_size_y=height,
                    annotation=annotation,
                    annotation_format=annotation_format,
                    image_path=image_path,  # Ссылка на оригинальное изображение
                    object_ids=object_ids
                )

                # Визуализация исходных аннотаций
                if VISUALIZE_SPLITS:
                    original_vis_path = ORIGINAL_ANNOTATIONS_DIR / f"{image_path.stem}_original{IMAGE_FORMAT}"
                    visualize_annotations(image, annotation, annotation_format, original_vis_path)

                try:
                    split_images_with_pos = multi_cut(annotated_image)
                except Exception as e:
                    logging.error(f"Ошибка при разрезании изображения '{image_path}': {e}")
                    continue

                # Отслеживание уникальных наборов object_id  - теперь храним frozenset
                unique_object_sets = set()
                duplicates = []

                # Визуализация разрезов
                if VISUALIZE_SPLITS:
                    split_coords_list = [item[-1] for item in split_images_with_pos]
                    splits_vis_path = SPLITS_DIR / f"{image_path.stem}_splits{IMAGE_FORMAT}"
                    if VISUALIZE_SPLITS_ON_ANNOTATIONS:
                        # Совмещаем разрезы и аннотации
                        visualize_splits(image, split_coords_list, splits_vis_path, annotations=annotation,
                                         annotation_format=annotation_format)
                    else:
                        visualize_splits(image, split_coords_list, splits_vis_path)

                for split_img_data, split_image_path, split_x, split_y, object_ids_in_crop, best_cut in split_images_with_pos:
                    object_ids_key = frozenset(object_ids_in_crop)
                    # Обновление пути для сохранения в соответствии с относительной структурой
                    split_image_path = image_output_dir / split_image_path.name
                    new_annotation_path = split_image_path.with_suffix(annotation_format)

                    is_duplicate = False

                    if object_ids_key in unique_object_sets and len(object_ids_key) > 0:
                        is_duplicate = True

                    if is_duplicate:
                        duplicates.append((split_img_data, split_image_path, split_x, split_y, new_annotation_path))
                        continue

                    if len(object_ids_key) > 0:  # Добавляем только непустые наборы
                        unique_object_sets.add(object_ids_key)

                    # Сохранение изображения
                    try:
                        save_image(split_img_data.data, split_image_path, IMAGE_FORMAT)
                    except IOError as e:
                        logging.error(e)
                        continue  # Переход к следующей части

                    # Сохранение аннотаций
                    if annotation_format == '.xml':
                        try:
                            new_tree = ET.ElementTree(split_img_data.annotation)
                            new_tree.write(new_annotation_path, encoding='utf-8', xml_declaration=True)
                            logging.debug(f"Сохранены аннотации XML: {new_annotation_path}")
                        except Exception as e:
                            logging.error(f"Ошибка при сохранении XML-аннотаций '{new_annotation_path}': {e}")
                    elif annotation_format == '.txt':
                        try:
                            with new_annotation_path.open('w', encoding='utf-8') as f:
                                for line in split_img_data.annotation:
                                    f.write(f"{line}\n")
                            logging.debug(f"Сохранены аннотации YOLO: {new_annotation_path}")
                        except Exception as e:
                            logging.error(f"Ошибка при сохранении YOLO-аннотаций '{new_annotation_path}': {e}")

                    # Визуализация аннотаций на подснимках
                    if VISUALIZE_SPLITS:
                        vis_image_path = ANNOTATED_SPLITS_DIR / f"{split_image_path.stem}_annotated{IMAGE_FORMAT}"
                        visualize_annotations(split_img_data.data, split_img_data.annotation, annotation_format, vis_image_path)

                    # Запись в CSV лог
                    log_writer.writerow([
                        str(image_path),
                        str(annotation_path),
                        str(split_image_path),
                        str(split_x),
                        str(split_y),
                        str(new_annotation_path)
                    ])

                # Обработка дубликатов
                for split_img_data, split_image_path, split_x, split_y, new_annotation_path in duplicates:
                    # Обновление пути для сохранения в директории дубликатов
                    split_image_path = duplicates_dir / split_image_path.name
                    new_annotation_path = split_image_path.with_suffix(annotation_format)

                    try:
                        save_image(split_img_data.data, split_image_path, IMAGE_FORMAT)
                    except IOError as e:
                        logging.error(e)
                        continue  # Переход к следующей части

                    if annotation_format == '.xml':

                        try:
                            new_tree = ET.ElementTree(split_img_data.annotation)
                            new_tree.write(new_annotation_path, encoding='utf-8', xml_declaration=True)
                            logging.debug(f"Сохранены аннотации XML (дубликат): {new_annotation_path}")
                        except Exception as e:
                            logging.error(f"Ошибка при сохранении XML-аннотаций '{new_annotation_path}': {e}")
                    elif annotation_format == '.txt':
                        # Сохранение YOLO
                        try:
                            with new_annotation_path.open('w', encoding='utf-8') as f:
                                for line in split_img_data.annotation:
                                    f.write(f"{line}\n")
                            logging.debug(f"Сохранены аннотации YOLO (дубликат): {new_annotation_path}")
                        except Exception as e:
                            logging.error(f"Ошибка при сохранении YOLO-аннотаций '{new_annotation_path}': {e}")

                    # Запись в CSV лог
                    log_writer.writerow([
                        str(image_path),
                        str(annotation_path),
                        str(split_image_path),
                        str(split_x),
                        str(split_y),
                        str(new_annotation_path)
                    ])

    except Exception as e:
        logging.error(f"Ошибка при инициализации или записи в CSV лог: {e}")
        sys.exit(1)

    logging.info("Обработка всех изображений завершена.")

def inverse_process_log(output_dir: Path) -> Dict[str, Dict]:
    """
    Обрабатывает лог-файл и собирает из аннотаций частей исходную аннотацию.

    :param output_dir: Путь к директории с результатами и логом.
    :return: Словарь с восстановленными аннотациями для каждого оригинального изображения.
    """
    reconstructed_annotations = defaultdict(lambda: {'objects': {}})

    try:
        with LOG_CSV_PATH.open('r', newline='', encoding='utf-8') as csvfile:
            log_reader = csv.DictReader(csvfile)
            for row in log_reader:
                original_image = row['Original Image']
                split_image = row['Split Image']
                split_pos_x = int(row['Split Position X'])
                split_pos_y = int(row['Split Position Y'])
                split_annotation = row['Split Annotation']

                # Загрузка аннотаций частей
                split_annotation_path = Path(split_annotation)
                if split_annotation_path.suffix.lower() == '.xml':
                    try:
                        tree = ET.parse(split_annotation_path)
                        root = tree.getroot()
                        objects = root.findall("object")
                        for obj in objects:
                            bndbox = obj.find("bndbox")
                            xmin = int(bndbox.find("xmin").text) + split_pos_x
                            ymin = int(bndbox.find("ymin").text) + split_pos_y
                            xmax = int(bndbox.find("xmax").text) + split_pos_x
                            ymax = int(bndbox.find("ymax").text) + split_pos_y

                            object_id_elem = obj.find("object_id")
                            if object_id_elem is not None:
                                object_id = int(object_id_elem.text)
                            else:
                                continue  # Пропускаем объекты без object_id

                            # Проверяем, был ли объект уже добавлен
                            if object_id not in reconstructed_annotations[original_image]['objects']:
                                # Создаем новый объект с обновленными координатами
                                new_obj = ET.Element("object")
                                for elem in obj:
                                    if elem.tag == "bndbox":
                                        new_bndbox = ET.SubElement(new_obj, "bndbox")
                                        ET.SubElement(new_bndbox, "xmin").text = str(xmin)
                                        ET.SubElement(new_bndbox, "ymin").text = str(ymin)
                                        ET.SubElement(new_bndbox, "xmax").text = str(xmax)
                                        ET.SubElement(new_bndbox, "ymax").text = str(ymax)
                                    else:
                                        new_elem = ET.SubElement(new_obj, elem.tag)
                                        new_elem.text = elem.text
                                reconstructed_annotations[original_image]['objects'][object_id] = new_obj
                    except ET.ParseError as e:
                        logging.error(f"Ошибка парсинга XML аннотаций из '{split_annotation}': {e}")
                        continue
                elif split_annotation_path.suffix.lower() == '.txt':
                    try:
                        with split_annotation_path.open('r', encoding='utf-8') as f:
                            lines = f.readlines()
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) == 6:
                                    class_id, x_center, y_center, width, height, object_id = parts
                                    object_id = int(object_id)
                                elif len(parts) == 5:
                                    class_id, x_center, y_center, width, height = parts
                                    # Используем original_image при генерации object_id
                                    object_id = generate_unique_object_id(original_image, line)
                                else:
                                    continue  # Пропускаем некорректные строки

                                class_id = int(class_id)
                                x_center = float(x_center)
                                y_center = float(y_center)
                                width = float(width)
                                height = float(height)

                                # Извлекаем размеры вырезанного изображения из файла split_image
                                split_image_path = Path(split_image)
                                if not split_image_path.is_file():
                                    logging.warning(f"Split image file не найден: {split_image_path}")
                                    continue
                                split_image_cv = load_image(split_image_path)
                                if split_image_cv is None:
                                    logging.warning(f"Не удалось загрузить split image: {split_image_path}")
                                    continue
                                split_height, split_width = split_image_cv.shape[:2]

                                # Преобразование нормализованных координат в абсолютные координаты в разрезанном изображении
                                obj_x_center_split = x_center * split_width
                                obj_y_center_split = y_center * split_height
                                obj_width_split = width * split_width
                                obj_height_split = height * split_height

                                # Преобразование координат в систему координат исходного изображения
                                obj_x_center = obj_x_center_split + split_pos_x
                                obj_y_center = obj_y_center_split + split_pos_y

                                xmin = obj_x_center - obj_width_split / 2
                                ymin = obj_y_center - obj_height_split / 2
                                xmax = obj_x_center + obj_width_split / 2
                                ymax = obj_y_center + obj_height_split / 2

                                # Проверяем, был ли объект уже добавлен
                                if object_id not in reconstructed_annotations[original_image]['objects']:
                                    # Создаем новый объект в формате PASCAL VOC для удобства сравнения
                                    new_obj = ET.Element("object")
                                    ET.SubElement(new_obj, "name").text = str(class_id)
                                    ET.SubElement(new_obj, "pose").text = "Unspecified"
                                    ET.SubElement(new_obj, "truncated").text = "0"
                                    ET.SubElement(new_obj, "difficult").text = "0"
                                    ET.SubElement(new_obj, "object_id").text = str(object_id)
                                    bndbox = ET.SubElement(new_obj, "bndbox")
                                    ET.SubElement(bndbox, "xmin").text = str(int(xmin))
                                    ET.SubElement(bndbox, "ymin").text = str(int(ymin))
                                    ET.SubElement(bndbox, "xmax").text = str(int(xmax))
                                    ET.SubElement(bndbox, "ymax").text = str(int(ymax))
                                    reconstructed_annotations[original_image]['objects'][object_id] = new_obj
                    except Exception as e:
                        logging.error(f"Ошибка чтения YOLO аннотаций из '{split_annotation}': {e}")
                        continue
                else:
                    logging.warning(f"Неизвестный формат аннотаций: {split_annotation_path.suffix}")
                    continue
    except FileNotFoundError:
        logging.error(f"Лог-файл не найден: {LOG_CSV_PATH}")
    except Exception as e:
        logging.error(f"Ошибка при обработке лог-файла: {e}")

    return reconstructed_annotations

def compute_iou(boxA, boxB):
    # Координаты пересечения
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Площадь пересечения
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    # Площади прямоугольников
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def compare_annotations(original_dir: Path, reconstructed_annotations: Dict[str, Dict]) -> None:
    """
    Сравнивает исходные и восстановленные аннотации, выводя разницу в пикселях.

    :param original_dir: Путь к директории с исходными изображениями и аннотациями.
    :param reconstructed_annotations: Словарь с восстановленными аннотациями.
    """
    delta_results = []

    for original_image, data in reconstructed_annotations.items():
        original_annotation_path_xml = Path(original_image).with_suffix('.xml')
        original_annotation_path_txt = Path(original_image).with_suffix('.txt')
        if original_annotation_path_xml.exists():
            original_annotation_path = original_annotation_path_xml
        elif original_annotation_path_txt.exists():
            original_annotation_path = original_annotation_path_txt
        else:
            logging.warning(f"Исходный файл аннотаций не найден для изображения: {original_image}")
            continue

        if original_annotation_path.suffix.lower() == '.xml':
            try:
                # Загружаем исходные XML аннотации
                tree = ET.parse(str(original_annotation_path))
                root = tree.getroot()
                original_objects = root.findall("object")

                original_bboxes = {}
                for obj in original_objects:
                    object_id_elem = obj.find("object_id")
                    if object_id_elem is not None:
                        object_id = int(object_id_elem.text)
                    else:
                        continue  # Пропускаем объекты без object_id

                    bndbox = obj.find("bndbox")
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)
                    original_bboxes[object_id] = (xmin, ymin, xmax, ymax)

                # Восстановленные аннотации
                reconstructed_objects = data.get('objects', {})
                for object_id, rec_obj in reconstructed_objects.items():
                    if object_id in original_bboxes:
                        orig_bbox = original_bboxes[object_id]
                        bndbox = rec_obj.find("bndbox")
                        xmin = float(bndbox.find("xmin").text)
                        ymin = float(bndbox.find("ymin").text)
                        xmax = float(bndbox.find("xmax").text)
                        ymax = float(bndbox.find("ymax").text)
                        rec_bbox = (xmin, ymin, xmax, ymax)

                        delta_xmin = abs(orig_bbox[0] - rec_bbox[0])
                        delta_ymin = abs(orig_bbox[1] - rec_bbox[1])
                        delta_xmax = abs(orig_bbox[2] - rec_bbox[2])
                        delta_ymax = abs(orig_bbox[3] - rec_bbox[3])

                        delta_pixels = f"Delta_xmin: {delta_xmin:.2f}, Delta_ymin: {delta_ymin:.2f}, Delta_xmax: {delta_xmax:.2f}, Delta_ymax: {delta_ymax:.2f}"

                        delta_results.append({
                            'Original Image': original_image,
                            'Object ID': object_id,
                            'Delta Pixels': delta_pixels
                        })
                    else:
                        delta_results.append({
                            'Original Image': original_image,
                            'Object ID': object_id,
                            'Delta Pixels': "Object not found in original annotations"
                        })
            except Exception as e:
                logging.error(f"Ошибка обработки XML аннотаций для '{original_annotation_path}': {e}")
                continue

        elif original_annotation_path.suffix.lower() == '.txt':
            try:
                # Загружаем исходные YOLO аннотации
                with original_annotation_path.open('r', encoding='utf-8') as f:
                    original_yolo_annotations = [line.strip() for line in f if line.strip()]

                original_bboxes = {}
                for line in original_yolo_annotations:
                    parts = line.strip().split()
                    if len(parts) == 6:
                        class_id, x_center, y_center, width, height, object_id = parts
                        object_id = int(object_id)
                    elif len(parts) == 5:
                        class_id, x_center, y_center, width, height = parts
                        # Используем original_image_path при генерации object_id
                        original_image_path = Path(original_image)
                        object_id = generate_unique_object_id(str(original_image_path), line)
                    else:
                        continue

                    class_id = int(class_id)
                    x_center = float(x_center)
                    y_center = float(y_center)
                    width = float(width)
                    height = float(height)

                    # Загрузка изображения для получения размеров
                    original_image_path = Path(original_image)
                    image = load_image(original_image_path)
                    if image is None:
                        logging.warning(f"Не удалось загрузить изображение для аннотаций: {original_image_path}")
                        continue
                    img_height, img_width = image.shape[:2]
                    obj_x_center = x_center * img_width
                    obj_y_center = y_center * img_height
                    obj_width = width * img_width
                    obj_height = height * img_height
                    xmin = obj_x_center - obj_width / 2
                    ymin = obj_y_center - obj_height / 2
                    xmax = obj_x_center + obj_width / 2
                    ymax = obj_y_center + obj_height / 2

                    original_bboxes[object_id] = (xmin, ymin, xmax, ymax)

                # Восстановленные аннотации
                reconstructed_objects = data.get('objects', {})
                for object_id, rec_obj in reconstructed_objects.items():
                    if object_id in original_bboxes:
                        orig_bbox = original_bboxes[object_id]
                        bndbox = rec_obj.find("bndbox")
                        xmin = float(bndbox.find("xmin").text)
                        ymin = float(bndbox.find("ymin").text)
                        xmax = float(bndbox.find("xmax").text)
                        ymax = float(bndbox.find("ymax").text)
                        rec_bbox = (xmin, ymin, xmax, ymax)

                        delta_xmin = abs(orig_bbox[0] - rec_bbox[0])
                        delta_ymin = abs(orig_bbox[1] - rec_bbox[1])
                        delta_xmax = abs(orig_bbox[2] - rec_bbox[2])
                        delta_ymax = abs(orig_bbox[3] - rec_bbox[3])

                        delta_pixels = f"Delta_xmin: {delta_xmin:.2f}, Delta_ymin: {delta_ymin:.2f}, Delta_xmax: {delta_xmax:.2f}, Delta_ymax: {delta_ymax:.2f}"

                        delta_results.append({
                            'Original Image': original_image,
                            'Object ID': object_id,
                            'Delta Pixels': delta_pixels
                        })
                    else:
                        delta_results.append({
                            'Original Image': original_image,
                            'Object ID': object_id,
                            'Delta Pixels': "Object not found in original annotations"
                        })
            except Exception as e:
                logging.error(f"Ошибка обработки YOLO аннотаций для '{original_annotation_path}': {e}")
                continue
        else:
            logging.warning(f"Неизвестный формат аннотаций: {original_annotation_path.suffix}")
            continue

    # Запись результатов сравнения в CSV
    try:
        with DELTA_PIXELS_CSV.open('w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Original Image', 'Object ID', 'Delta Pixels']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in delta_results:
                writer.writerow(row)
        logging.info(f"Сравнение аннотаций завершено. Результаты сохранены в '{DELTA_PIXELS_CSV}'.")
    except Exception as e:
        logging.error(f"Ошибка при записи результатов сравнения в CSV: {e}")

def remove_object_id_from_xml_files(output_dir: Path):
    """
    Удаляет элементы 'object_id' из всех XML-аннотаций в заданной директории рекурсивно.

    :param output_dir: Директория, в которой необходимо удалить 'object_id' из XML-файлов.
    """
    for xml_file in output_dir.rglob('*.xml'):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                object_id_elem = obj.find('object_id')
                if object_id_elem is not None:
                    obj.remove(object_id_elem)
            tree.write(xml_file, encoding='utf-8', xml_declaration=True)
            logging.debug(f"Удалено 'object_id' из файла: {xml_file}")
        except Exception as e:
            logging.error(f"Ошибка при обработке файла '{xml_file}': {e}")

def remove_object_id_from_txt_files(output_dir: Path):
    """
    Удаляет 'object_id' из всех TXT-аннотаций в заданной директории рекурсивно.

    :param output_dir: Директория, в которой необходимо удалить 'object_id' из TXT-файлов.
    """
    for txt_file in output_dir.rglob('*.txt'):
        try:
            lines = txt_file.read_text(encoding='utf-8').splitlines()
            updated_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 6:
                    # Удаляем последнее поле (object_id)
                    parts = parts[:-1]
                    updated_line = ' '.join(parts)
                    updated_lines.append(updated_line)
                else:
                    # Если object_id нет, оставляем строку без изменений
                    updated_lines.append(line.strip())
            txt_file.write_text('\n'.join(updated_lines), encoding='utf-8')
            logging.debug(f"Удалено 'object_id' из файла: {txt_file}")
        except Exception as e:
            logging.error(f"Ошибка при обработке файла '{txt_file}': {e}")

def main():
    """
    Основная функция скрипта.
    """
    # Настройка логирования
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

    # Проверка наличия директории с изображениями
    if not INPUT_DIR.is_dir():
        logging.error(f"Входная директория не существует или не является директорией: {INPUT_DIR}")
        sys.exit(1)

    # Создание выходной директории, если она не существует
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Не удалось создать выходную директорию '{OUTPUT_DIR}': {e}")
        sys.exit(1)

    # Вызов функции обработки всех изображений
    process_all_images(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR
    )

    # Обратная обработка лога для восстановления аннотаций
    reconstructed_annotations = inverse_process_log(OUTPUT_DIR)

    # Сравнение исходных и восстановленных аннотаций
    compare_annotations(INPUT_DIR, reconstructed_annotations)

    # Удаление 'object_id' из всех аннотаций после сравнения
    remove_object_id_from_xml_files(OUTPUT_DIR)
    remove_object_id_from_txt_files(OUTPUT_DIR)

if __name__ == "__main__":
    main()
