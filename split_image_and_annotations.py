import cv2
import xml.etree.ElementTree as ET
import logging
from typing import Tuple, List, Optional, Dict, Union
from pathlib import Path
import sys
import numpy as np
import csv
from collections import defaultdict
import re
from enum import Enum

# --------------------------- Константы ---------------------------

# Путь к директории с исходными данными (изображениями и аннотациями)
# NOTE поставил свои пути
INPUT_DIR = Path(r"C:\Users\mkolp\OneDrive\Изображения\test")

# Путь к директории для сохранения результатов
# NOTE поставил свои пути
OUTPUT_DIR = Path(r"C:\Users\mkolp\OneDrive\Изображения\test\results")

# Параметры разделения изображения
SPLIT_HORIZONTAL = 2  # Количество разбиений по горизонтали
SPLIT_VERTICAL = 2    # Количество разбиений по вертикали
OVERLAP = 0.4 # перекрытие двух соседних частей [долей от исходного]

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
                 image_path: Optional[Path] = None):
        super().__init__(data, image_size_x, image_size_y)
        self.annotation = annotation
        self.annotation_format = annotation_format  # '.xml' или '.txt'
        self.image_path = image_path  # Путь к текущему изображению

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
    coords = []
    sub_width = int(width / (split_h - (split_h - 1) * overlap))
    sub_height = int(height / (split_v - (split_v - 1) * overlap))
    step_x = int(sub_width * (1 - overlap))
    step_y = int(sub_height * (1 - overlap))

    for i in range(split_v):
        for j in range(split_h):
            x1 = int(j * step_x)
            y1 = int(i * step_y)
            x2 = x1 + sub_width
            y2 = y1 + sub_height

            # Корректируем координаты, чтобы не выходили за границы изображения
            x2 = min(x2, width)
            y2 = min(y2, height)

            coords.append((x1, y1, x2, y2))

    return coords


def adjust_bndbox_voc(bndbox: ET.Element, split_coords: Tuple[int, int, int, int],
                      cropped_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    """
    Корректирует координаты ограничивающего прямоугольника объекта относительно вырезанного участка для PASCAL VOC (XML).

    :param bndbox: Элемент XML с координатами объекта.
    :param split_coords: Кортеж с координатами участка (x1, y1, x2, y2).
    :param cropped_size: Размеры вырезанного изображения (width, height).
    :return: Кортеж с новыми координатами (new_xmin, new_ymin, new_xmax, new_ymax) или None.
    """
    x1, y1, x2, y2 = split_coords
    cropped_width, cropped_height = cropped_size

    try:
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
    except (AttributeError, ValueError) as e:
        logging.warning(f"Некорректные координаты bndbox: {e}")
        return None

    # Проверка пересечения
    if xmax <= x1 or xmin >= x2 or ymax <= y1 or ymin >= y2:
        return None  # Объект не пересекается с текущим участком

    # Проверяем, полностью ли объект внутри разреза
    if xmin >= x1 and xmax <= x2 and ymin >= y1 and ymax <= y2:
        # Объект полностью внутри разреза, корректируем координаты относительно вырезанного участка
        return (xmin - x1, ymin - y1, xmax - x1, ymax - y1)  # Полностью внутри

    # Корректировка координат
    new_xmin = max(xmin - x1, 0)
    new_ymin = max(ymin - y1, 0)
    new_xmax = min(xmax - x1, cropped_width)
    new_ymax = min(ymax - y1, cropped_height)

    # Проверка минимальной площади пересечения
    if (new_xmax - new_xmin) * (new_ymax - new_ymin) < MIN_INTERSECTION_AREA:
        return None

    if new_xmax <= new_xmin or new_ymax <= new_ymin:
        return None

    return new_xmin, new_ymin, new_xmax, new_ymax


def adjust_bndbox_yolo(bndbox: str, split_coords: Tuple[int, int, int, int],
                      original_size: Tuple[int, int],
                      cropped_size: Tuple[int, int]) -> Optional[str]:
    """
    Корректирует координаты ограничивающего прямоугольника объекта относительно вырезанного участка для YOLO (TXT).

    :param bndbox: Строка с координатами объекта в формате YOLO (class x_center y_center width height).
    :param split_coords: Кортеж с координатами участка (x1, y1, x2, y2).
    :param original_size: Размеры исходного изображения (width, height).
    :param cropped_size: Размеры вырезанного изображения (width, height).
    :return: Строка с новыми координатами объекта или None.
    """
    try:
        parts = bndbox.strip().split()
        if len(parts) != 5:
            logging.warning(f"Некорректный формат bndbox YOLO: {bndbox}")
            return None
        class_id, x_center, y_center, width, height = map(float, parts)
    except ValueError as e:
        logging.warning(f"Некорректные координаты bndbox YOLO: {e}")
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

    return f"{int(class_id)} {norm_x_center:.6f} {norm_y_center:.6f} {norm_width:.6f} {norm_height:.6f}"


def cut_with_annotation(annotated_image: AnnotatedImageData, cut_set: Tuple[int, int, int, int],
                        split_idx: int) -> Tuple[AnnotatedImageData, Path, int, int]:
    """
    Вырезает часть изображения и соответствующие аннотации.

    :param annotated_image: Объект AnnotatedImageData с исходными данными.
    :param cut_set: Кортеж с координатами разреза (x1, y1, x2, y2).
    :param split_idx: Индекс текущей части для имени файла.
    :return: Кортеж из нового AnnotatedImageData с вырезанным изображением и обновленными аннотациями,
             и пути к новому изображению, а также позиций левого верхнего угла (x1, y1).
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
    new_image_path = annotated_image.image_path.parent / new_filename if annotated_image.image_path else Path(new_filename)

    # Корректируем аннотации
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
            adjusted_coords = adjust_bndbox_voc(bndbox, cut_set, (cropped_width, cropped_height))
            if adjusted_coords is None:
                continue  # Объект не пересекается или пересечение слишком мало

            new_obj = ET.SubElement(new_annotation_root, "object")
            # Копирование всех подэлементов объекта кроме bndbox
            for elem in obj:
                if elem.tag != "bndbox":
                    new_elem = ET.SubElement(new_obj, elem.tag)
                    new_elem.text = elem.text

            # Создание нового bndbox
            new_bndbox = ET.SubElement(new_obj, "bndbox")
            ET.SubElement(new_bndbox, "xmin").text = str(adjusted_coords[0])
            ET.SubElement(new_bndbox, "ymin").text = str(adjusted_coords[1])
            ET.SubElement(new_bndbox, "xmax").text = str(adjusted_coords[2])
            ET.SubElement(new_bndbox, "ymax").text = str(adjusted_coords[3])

        new_annotation = new_annotation_root

    elif annotated_image.annotation_format == '.txt':
        new_yolo_annotations = []
        original_size = (annotated_image.image_size_x, annotated_image.image_size_y)
        for line in annotated_image.annotation:
            adjusted_ann = adjust_bndbox_yolo(line, cut_set, original_size, (cropped_width, cropped_height))
            if adjusted_ann is not None:
                new_yolo_annotations.append(adjusted_ann)

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
        image_path=new_image_path  # Устанавливаем путь к новому изображению
    )

    return split_image_data, new_image_path, x1, y1


def verify(annotated_image: AnnotatedImageData, cut_set: Tuple[int, int, int, int]) -> Tuple[int, float, Tuple[bool, bool]]:
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
            if len(parts) != 5:
                continue
            try:
                class_id, x_center, y_center, width, height = map(float, parts)
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


def multi_cut(annotated_image: AnnotatedImageData) -> List[Tuple[AnnotatedImageData, Path, int, int]]:
    """
    Разделяет изображение и аннотации на 2x2 сетку.

    :param annotated_image: Объект AnnotatedImageData с исходными данными.
    :return: Список кортежей из AnnotatedImageData с вырезанными частями, пути к новым изображениям и позиций (x1, y1).
    """
    # Вычисление базовых координат разреза
    split_coords_list = calculate_split_coordinates(
        annotated_image.image_size_x,
        annotated_image.image_size_y,
        SPLIT_HORIZONTAL,
        SPLIT_VERTICAL,
        OVERLAP # NOTE перекрытие
    )
    total_parts = len(split_coords_list)
    logging.info(f"Разделение изображения на {total_parts} частей.")

    # Параметры сдвига
    step = 15  # шаг смещения в пикселях
    nstep = 10  # количество попыток сдвига

    image_result = []  # список вырезанных изображений и их позиций

    # Возможные направления сдвига для каждой части (право-вниз, влево-вниз, право-вверх, влево-вверх)
    directions = [
        (step, step),    # часть 1: вправо и вниз
        (-step, step),   # часть 2: влево и вниз
        (step, -step),   # часть 3: вправо и вверх
        (-step, -step)   # часть 4: влево и вверх
    ]

    for idx, base_split in enumerate(split_coords_list, start=1):
        logging.info(f"Обработка части {idx} из {total_parts} (Координаты: {base_split})...")
        a_base, b_base, c_base, d_base = base_split

        best_cut = base_split
        best_min_percent = 0.0
        best_N = float('inf')
        best_axis = (False, False)
        done = False
        results_cut: Dict[Tuple[int, int, int, int], Tuple[int, float, Tuple[bool, bool]]] = {}

        for attempt in range(nstep):
            if attempt == 0:
                da, dc = 0, 0
            else:
                # Используем best_axis для определения направлений смещения
                da = step * attempt if best_axis[0] else 0
                dc = step * attempt if best_axis[1] else 0

            # Смещение разреза только по необходимым осям
            current_split = (
                a_base + da,
                b_base + dc,
                c_base + da,
                d_base + dc
            )

            # Проверяем границы, чтобы не выйти за пределы изображения
            current_split = (
                max(current_split[0], 0),
                max(current_split[1], 0),
                min(current_split[2], annotated_image.image_size_x),
                min(current_split[3], annotated_image.image_size_y)
            )

            # Проверяем, что разрез корректен
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
                results_cut[current_split] = (N, min_percent, axis)
                if min_percent > best_min_percent:
                    best_min_percent = min_percent
                    best_cut = current_split
                    best_N = N
                    best_axis = axis

        if not done:
            logging.info(f"Идеальный разрез не найден для части {idx}. Выбирается наилучший из худших вариантов.")

        # Вырезаем изображение и аннотации по выбранному разрезу
        split_image_data, split_image_path, split_x, split_y = cut_with_annotation(annotated_image, best_cut, idx)
        image_result.append((split_image_data, split_image_path, split_x, split_y))

    return image_result


def process_all_images(input_dir: Path, output_dir: Path) -> None:
    """
    Обрабатывает все изображения и аннотации в указанной директории и её подпапках.

    :param input_dir: Путь к директории с исходными изображениями и аннотациями.
    :param output_dir: Путь к директории для сохранения разделенных изображений и аннотаций.
    """
    if not input_dir.is_dir():
        logging.error(f"Входная директория не существует или не является директорией: {input_dir}")
        sys.exit(1)

    # Поиск всех изображений в директории и подпапках
    image_files = [p for p in input_dir.rglob('*') if p.suffix.lower() in SUPPORTED_IMAGE_FORMATS]

    if not image_files:
        logging.warning(f"Входная директория '{input_dir}' не содержит поддерживаемых изображений.")
        return

    logging.info(f"Найдено {len(image_files)} изображений для обработки.")

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

                # Определение относительного пути к изображению для сохранения структуры
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
                if annotation_format == '.xml':
                    logging.info(f"Парсинг XML-аннотаций из '{annotation_path}'...")
                    try:
                        tree = ET.parse(str(annotation_path))
                        root = tree.getroot()
                        annotation = root
                    except ET.ParseError as e:
                        logging.error(f"Ошибка парсинга XML-аннотаций: {e}")
                        continue
                elif annotation_format == '.txt':
                    logging.info(f"Парсинг YOLO-аннотаций из '{annotation_path}'...")
                    try:
                        yolo_annotations = [line.strip() for line in
                                            annotation_path.read_text(encoding='utf-8').splitlines() if line.strip()]
                        annotation = yolo_annotations  # Список строк для корректной передачи в AnnotatedImageData
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
                logging.debug(f"Размер изображения: {width}x{height}, Глубина: {depth}")

                # Создание AnnotatedImageData объекта
                annotated_image = AnnotatedImageData(
                    data=image,
                    image_size_x=width,
                    image_size_y=height,
                    annotation=annotation,
                    annotation_format=annotation_format,
                    image_path=image_path  # Ссылка на оригинальное изображение
                )

                # Применение multi_cut
                try:
                    split_images_with_pos = multi_cut(annotated_image)
                except Exception as e:
                    logging.error(f"Ошибка при разрезании изображения '{image_path}': {e}")
                    continue

                # Сохранение разрезанных изображений и аннотаций
                for split_img_data, split_image_path, split_x, split_y in split_images_with_pos:
                    # Обновление пути для сохранения в соответствии с относительной структурой
                    split_image_path = image_output_dir / split_image_path.name

                    # Сохранение изображения
                    try:
                        save_image(split_img_data.data, split_image_path, IMAGE_FORMAT)
                    except IOError as e:
                        logging.error(e)
                        continue  # Переход к следующей части

                    # Сохранение аннотаций
                    new_annotation_path = split_image_path.with_suffix(annotation_format)

                    if annotation_format == '.xml':
                        # Сохранение XML
                        try:
                            new_tree = ET.ElementTree(split_img_data.annotation)
                            new_tree.write(new_annotation_path, encoding='utf-8', xml_declaration=True)
                            logging.debug(f"Сохранены аннотации XML: {new_annotation_path}")
                        except Exception as e:
                            logging.error(f"Ошибка при сохранении XML-аннотаций '{new_annotation_path}': {e}")
                    elif annotation_format == '.txt':
                        # Сохранение YOLO
                        try:
                            with new_annotation_path.open('w', encoding='utf-8') as f:
                                for line in split_img_data.annotation:
                                    f.write(f"{line}\n")
                            logging.debug(f"Сохранены аннотации YOLO: {new_annotation_path}")
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
    reconstructed_annotations = defaultdict(lambda: {'objects': []})

    try:
        with LOG_CSV_PATH.open('r', newline='', encoding='utf-8') as csvfile:
            log_reader = csv.DictReader(csvfile)
            for row in log_reader:
                original_image = row['Original Image']
                original_annotation = row['Original Annotation']
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
                            reconstructed_annotations[original_image]['objects'].append(new_obj)
                    except ET.ParseError as e:
                        logging.error(f"Ошибка парсинга XML аннотаций из '{split_annotation}': {e}")
                        continue
                elif split_annotation_path.suffix.lower() == '.txt':
                    try:
                        with split_annotation_path.open('r', encoding='utf-8') as f:
                            lines = f.readlines()
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) != 5:
                                    continue
                                class_id, x_center, y_center, width, height = map(float, parts)
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

                                # Создаем новый объект в формате PASCAL VOC для удобства сравнения
                                new_obj = ET.Element("object")
                                ET.SubElement(new_obj, "name").text = str(int(class_id))
                                ET.SubElement(new_obj, "pose").text = "Unspecified"
                                ET.SubElement(new_obj, "truncated").text = "0"
                                ET.SubElement(new_obj, "difficult").text = "0"
                                bndbox = ET.SubElement(new_obj, "bndbox")
                                ET.SubElement(bndbox, "xmin").text = str(int(xmin))
                                ET.SubElement(bndbox, "ymin").text = str(int(ymin))
                                ET.SubElement(bndbox, "xmax").text = str(int(xmax))
                                ET.SubElement(bndbox, "ymax").text = str(int(ymax))

                                reconstructed_annotations[original_image]['objects'].append(new_obj)
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
            # Существующий код для XML аннотаций (без изменений)
            # ...
            pass  # Оставляем существующий код
        elif original_annotation_path.suffix.lower() == '.txt':
            try:
                # Загружаем исходные YOLO аннотации
                with original_annotation_path.open('r', encoding='utf-8') as f:
                    original_yolo_annotations = [line.strip() for line in f if line.strip()]

                # Преобразуем исходные аннотации в абсолютные координаты
                original_bboxes = []
                for line in original_yolo_annotations:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    class_id, x_center, y_center, width, height = map(float, parts)
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
                    original_bboxes.append((xmin, ymin, xmax, ymax))

                # Восстановленные аннотации уже в формате PASCAL VOC
                reconstructed_objects = data.get('objects', [])
                reconstructed_bboxes = []
                for obj in reconstructed_objects:
                    bndbox = obj.find("bndbox")
                    if bndbox is not None:
                        xmin = float(bndbox.find("xmin").text)
                        ymin = float(bndbox.find("ymin").text)
                        xmax = float(bndbox.find("xmax").text)
                        ymax = float(bndbox.find("ymax").text)
                        reconstructed_bboxes.append((xmin, ymin, xmax, ymax))

                # Сопоставление на основе IoU
                matched_indices = set()
                for orig_idx, orig_bbox in enumerate(original_bboxes):
                    max_iou = 0
                    matched_rec_idx = -1
                    for rec_idx, rec_bbox in enumerate(reconstructed_bboxes):
                        if rec_idx in matched_indices:
                            continue
                        iou = compute_iou(orig_bbox, rec_bbox)
                        if iou > max_iou:
                            max_iou = iou
                            matched_rec_idx = rec_idx
                    if max_iou > 0.5 and matched_rec_idx != -1:
                        rec_bbox = reconstructed_bboxes[matched_rec_idx]
                        matched_indices.add(matched_rec_idx)
                        delta_xmin = abs(orig_bbox[0] - rec_bbox[0])
                        delta_ymin = abs(orig_bbox[1] - rec_bbox[1])
                        delta_xmax = abs(orig_bbox[2] - rec_bbox[2])
                        delta_ymax = abs(orig_bbox[3] - rec_bbox[3])
                        delta_pixels = f"Delta_xmin: {delta_xmin:.2f}, Delta_ymin: {delta_ymin:.2f}, Delta_xmax: {delta_xmax:.2f}, Delta_ymax: {delta_ymax:.2f}"
                    else:
                        delta_pixels = "No matching reconstructed object found"

                    delta_results.append({
                        'Original Image': original_image,
                        'Object Index': orig_idx,
                        'Delta Pixels': delta_pixels
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
            fieldnames = ['Original Image', 'Object Index', 'Delta Pixels']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in delta_results:
                writer.writerow(row)
        logging.info(f"Сравнение аннотаций завершено. Результаты сохранены в '{DELTA_PIXELS_CSV}'.")
    except Exception as e:
        logging.error(f"Ошибка при записи результатов сравнения в CSV: {e}")


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


if __name__ == "__main__":
    main()
