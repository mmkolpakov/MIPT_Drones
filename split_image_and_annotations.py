import cv2
import xml.etree.ElementTree as ET
import logging
from typing import Tuple, List, Optional, Dict, Union
from pathlib import Path
import sys
import numpy as np

# --------------------------- Константы ---------------------------

# Путь к директории с исходными данными (изображениями и аннотациями)
INPUT_DIR = Path(r"C:\Users\mkolp\Downloads\Batches")

# Путь к директории для сохранения результатов
OUTPUT_DIR = Path(r"C:\Users\mkolp\Downloads\Images")

# Параметры разделения изображения
SPLIT_HORIZONTAL = 2  # Количество разбиений по горизонтали
SPLIT_VERTICAL = 2    # Количество разбиений по вертикали

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
        self.image_path = image_path


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


def calculate_split_coordinates(width: int, height: int, split_h: int, split_v: int) -> List[Tuple[int, int, int, int]]:
    """
    Вычисляет координаты для разбиения изображения на сетку.

    :param width: Ширина изображения.
    :param height: Высота изображения.
    :param split_h: Количество разбиений по горизонтали.
    :param split_v: Количество разбиений по вертикали.
    :return: Список кортежей с координатами (x1, y1, x2, y2) для каждого участка.
    """
    coords = []
    step_x = width // split_h
    step_y = height // split_v

    for i in range(split_v):
        for j in range(split_h):
            x1 = j * step_x
            y1 = i * step_y
            # Для последнего разбиения по горизонтали и вертикали берем остаток
            x2 = (j + 1) * step_x if j < split_h - 1 else width
            y2 = (i + 1) * step_y if i < split_v - 1 else height
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
        # Объект полностью внутри разреза, не увеличиваем N
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
                        split_idx: int) -> AnnotatedImageData:
    """
    Вырезает часть изображения и соответствующие аннотации.

    :param annotated_image: Объект AnnotatedImageData с исходными данными.
    :param cut_set: Кортеж с координатами разреза (x1, y1, x2, y2).
    :param split_idx: Индекс текущей части для имени файла.
    :return: Новый объект AnnotatedImageData с вырезанным изображением и обновленными аннотациями.
    """
    x1, y1, x2, y2 = cut_set  # Извлечение координат

    # Вырезаем изображение
    cropped_image = annotated_image.data[y1:y2, x1:x2]
    cropped_height, cropped_width = cropped_image.shape[:2]

    # Генерация нового имени файла на основе имени изображения
    base_filename = Path(annotated_image.image_path).stem if annotated_image.image_path else 'image'
    new_filename = f"{base_filename}_part_{split_idx}{IMAGE_FORMAT}"

    # Корректируем аннотации
    if annotated_image.annotation_format == '.xml':
        new_annotation_root = ET.Element("annotation")

        # Копирование всех необходимых элементов из исходной аннотации
        # Исключаем только те, которые нужно изменить (filename, size, object)
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

    return AnnotatedImageData(
        data=cropped_image,
        image_size_x=cropped_width,
        image_size_y=cropped_height,
        annotation=new_annotation,
        annotation_format=annotated_image.annotation_format,
        image_path=annotated_image.image_path  # Сохранение пути изображения
    )


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
                # Определяем, полностью ли объект внутри разреза
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
                # Определяем, полностью ли объект внутри разреза
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


def multi_cut(annotated_image: AnnotatedImageData) -> List[AnnotatedImageData]:
    """
    Разделяет изображение и аннотации на сетку с учётом оптимального разреза.

    :param annotated_image: Объект AnnotatedImageData с исходными данными.
    :return: Список объектов AnnotatedImageData с вырезанными частями.
    """
    # Вычисление базовых координат разреза
    split_coords_list = calculate_split_coordinates(
        annotated_image.image_size_x,
        annotated_image.image_size_y,
        SPLIT_HORIZONTAL,
        SPLIT_VERTICAL
    )
    total_parts = len(split_coords_list)
    logging.info(f"Разделение изображения на {total_parts} частей.")

    # Параметры сдвига
    step = 15  # шаг смещения в пикселях
    nstep = 10  # количество попыток сдвига

    image_result = []  # список вырезанных изображений

    # Возможные направления сдвига для каждой части
    directions = [
        (step, step),    # часть 1: вправо и вниз
        (-step, step),   # часть 2: влево и вниз
        (step, -step),   # часть 3: вправо и вверх
        (-step, -step)   # часть 4: влево и вверх
    ]

    for idx, base_split in enumerate(split_coords_list, start=1):
        logging.info(f"Обработка части {idx} из {total_parts} (Координаты: {base_split})...")
        a_base, b_base, c_base, d_base = base_split

        if idx <= len(directions):
            da, dc = directions[idx - 1]
        else:
            da, dc = (step, step)  # По умолчанию

        best_cut = base_split
        best_min_percent = 0.0
        best_N = float('inf')
        best_axis = (False, False)
        done = False
        results_cut: Dict[Tuple[int, int, int, int], Tuple[int, float, Tuple[bool, bool]]] = {}

        for attempt in range(nstep):
            # Вырезаем изображение с текущими координатами
            current_split = (
                a_base + da * attempt,          # x1 сдвиг по dx
                b_base + dc * attempt,          # y1 сдвиг по dy
                c_base + da * attempt,          # x2 сдвиг по dx
                d_base + dc * attempt           # y2 сдвиг по dy
            )

            # Проверяем границы, чтобы не выйти за пределы изображения
            current_split = (
                max(current_split[0], 0),
                max(current_split[1], 0),
                min(current_split[2], annotated_image.image_size_x),
                min(current_split[3], annotated_image.image_size_y)
            )

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
            # Выбираем разрез с максимальной минимальной долей внутри
            for split_set, (N, min_percent, axis) in results_cut.items():
                if min_percent > best_min_percent:
                    best_min_percent = min_percent
                    best_cut = split_set
                    best_N = N
                    best_axis = axis

        # Вырезаем изображение и аннотации по выбранному разрезу
        cut_with_annotation_result = cut_with_annotation(annotated_image, best_cut, idx)
        image_result.append(cut_with_annotation_result)

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
            logging.error(f"Изображение '{image_path}' не находится внутри входной директории '{input_dir}'. Пропуск.")
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
                yolo_annotations = [line.strip() for line in annotation_path.read_text(encoding='utf-8').splitlines() if line.strip()]
                annotation = yolo_annotations  # Изменение для корректной передачи в AnnotatedImageData
            except Exception as e:
                logging.error(f"Ошибка чтения YOLO-аннотаций: {e}")
                continue
        else:
            logging.error(f"Формат аннотаций '{annotation_format}' не поддерживается.")
            continue

        # Загрузка изображения
        image = cv2.imread(str(image_path))
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
            image_path=image_path  # Добавлено
        )
        # Сохранение пути аннотации для использования в multi_cut
        # Уже сохранено через image_path

        # Применение multi_cut
        try:
            split_images = multi_cut(annotated_image)
        except Exception as e:
            logging.error(f"Ошибка при разрезании изображения '{image_path}': {e}")
            continue

        # Сохранение разрезанных изображений и аннотаций
        for idx, split_img in enumerate(split_images, start=1):
            new_filename = f"{Path(split_img.image_path).stem}_part_{idx}{IMAGE_FORMAT}"
            new_image_path = image_output_dir / new_filename

            # Сохранение изображения
            try:
                save_image(split_img.data, new_image_path, IMAGE_FORMAT)
            except IOError as e:
                logging.error(e)
                continue  # Переход к следующей части

            # Сохранение аннотаций
            new_annotation_filename = f"{Path(split_img.image_path).stem}_part_{idx}{annotation_format}"
            new_annotation_path = image_output_dir / new_annotation_filename

            if annotation_format == '.xml':
                # Сохранение XML
                try:
                    new_tree = ET.ElementTree(split_img.annotation)
                    new_tree.write(new_annotation_path, encoding='utf-8', xml_declaration=True)
                    logging.debug(f"Сохранены аннотации XML: {new_annotation_path}")
                except Exception as e:
                    logging.error(f"Ошибка при сохранении XML-аннотаций '{new_annotation_path}': {e}")
            elif annotation_format == '.txt':
                # Сохранение YOLO
                try:
                    with new_annotation_path.open('w', encoding='utf-8') as f:
                        for line in split_img.annotation:
                            f.write(f"{line}\n")
                    logging.debug(f"Сохранены аннотации YOLO: {new_annotation_path}")
                except Exception as e:
                    logging.error(f"Ошибка при сохранении YOLO-аннотаций '{new_annotation_path}': {e}")

    logging.info("Обработка всех изображений завершена.")


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


if __name__ == "__main__":
    main()
