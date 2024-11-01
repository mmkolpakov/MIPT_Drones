import cv2
import xml.etree.ElementTree as ET
import logging
from typing import Tuple, List, Optional
from pathlib import Path
import sys

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


def save_image(img: any, output_path: Path, image_format: str) -> None:
    """
    Сохраняет изображение с поддержкой путей с нелатинскими символами.

    :param img: Изображение в формате NumPy массива.
    :param output_path: Путь для сохранения изображения.
    :param image_format: Формат изображения (например, '.jpg').
    :raises IOError: Если сохранение изображения не удалось.
    """
    try:
        success, encoded_image = cv2.imencode(image_format, img)
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


def adjust_bndbox_voc(bndbox: ET.Element, split_coords: Tuple[int, int, int, int], cropped_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
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


def adjust_bndbox_yolo(bndbox: str, split_coords: Tuple[int, int, int, int], cropped_size: Tuple[int, int]) -> Optional[str]:
    """
    Корректирует координаты ограничивающего прямоугольника объекта относительно вырезанного участка для YOLO (TXT).

    :param bndbox: Строка с координатами объекта в формате YOLO (class x_center y_center width height).
    :param split_coords: Кортеж с координатами участка (x1, y1, x2, y2).
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

    # Преобразуем координаты из YOLO в xmin, ymin, xmax, ymax
    img_width, img_height = split_coords[2] - split_coords[0], split_coords[3] - split_coords[1]
    x_center_abs = x_center * img_width + split_coords[0]
    y_center_abs = y_center * img_height + split_coords[1]
    width_abs = width * img_width
    height_abs = height * img_height

    xmin = x_center_abs - width_abs / 2
    ymin = y_center_abs - height_abs / 2
    xmax = x_center_abs + width_abs / 2
    ymax = y_center_abs + height_abs / 2

    # Проверка пересечения с участком
    split_x1, split_y1, split_x2, split_y2 = split_coords
    if xmax <= split_x1 or xmin >= split_x2 or ymax <= split_y1 or ymin >= split_y2:
        return None  # Объект не пересекается

    # Корректировка координат относительно вырезанного участка
    new_xmin = max(xmin - split_x1, 0)
    new_ymin = max(ymin - split_y1, 0)
    new_xmax = min(xmax - split_x1, cropped_size[0])
    new_ymax = min(ymax - split_y1, cropped_size[1])

    # Проверка минимальной площади пересечения
    if (new_xmax - new_xmin) * (new_ymax - new_ymin) < MIN_INTERSECTION_AREA:
        return None

    if new_xmax <= new_xmin or new_ymax <= new_ymin:
        return None

    # Преобразуем обратно в YOLO формат
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


def split_image_and_annotations(
        image_path: Path,
        annotation_path: Path,
        output_dir: Path,
        split_h: int = SPLIT_HORIZONTAL,
        split_v: int = SPLIT_VERTICAL
) -> None:
    """
    Разделяет изображение и соответствующие аннотации на сетку.

    :param image_path: Путь к исходному изображению.
    :param annotation_path: Путь к исходному файлу аннотаций.
    :param output_dir: Путь к директории для сохранения разделенных изображений и аннотаций.
    :param split_h: Количество разбиений по горизонтали.
    :param split_v: Количество разбиений по вертикали.
    """
    try:
        logging.info(f"Создание выходной директории для '{image_path.name}'...")
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Не удалось создать директорию '{output_dir}': {e}")
        return

    # Проверка формата изображения
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
        logging.error(
            f"Неподдерживаемый формат изображения '{image_path.suffix}'. Поддерживаются: {SUPPORTED_IMAGE_FORMATS}")
        return

    # Определение формата аннотации
    annotation_format = annotation_path.suffix.lower()
    if annotation_format not in SUPPORTED_ANNOTATION_FORMATS:
        logging.error(
            f"Неподдерживаемый формат аннотации '{annotation_format}'. Поддерживаются: {SUPPORTED_ANNOTATION_FORMATS}")
        return

    # Загрузка изображения
    logging.info(f"Загрузка изображения из '{image_path}'...")
    image = cv2.imread(str(image_path))
    if image is None:
        logging.error(f"Изображение не найдено или не может быть загружено: {image_path}")
        return
    height, width = image.shape[:2]
    depth = image.shape[2] if len(image.shape) == 3 else 1
    logging.debug(f"Размер изображения: {width}x{height}, Глубина: {depth}")

    # Парсинг аннотаций
    if annotation_format == '.xml':
        logging.info(f"Парсинг XML-аннотаций из '{annotation_path}'...")
        try:
            tree = ET.parse(str(annotation_path))
            root = tree.getroot()
        except ET.ParseError as e:
            logging.error(f"Ошибка парсинга XML-аннотаций: {e}")
            return
    elif annotation_format == '.txt':
        logging.info(f"Парсинг YOLO-аннотаций из '{annotation_path}'...")
        try:
            with annotation_path.open('r') as f:
                yolo_annotations = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logging.error(f"Ошибка чтения YOLO-аннотаций: {e}")
            return
    else:
        logging.error(f"Формат аннотаций '{annotation_format}' не поддерживается.")
        return

    # Вычисление координат разбиения
    coords = calculate_split_coordinates(width, height, split_h, split_v)
    total_parts = len(coords)
    logging.info(f"Изображение '{image_path.name}' будет разделено на {total_parts} частей.")

    base_filename = image_path.stem

    for idx, split_coord in enumerate(coords, start=1):
        x1, y1, x2, y2 = split_coord
        logging.info(f"Обработка части {idx} из {total_parts} (Координаты: {split_coord})...")
        cropped_image = image[y1:y2, x1:x2]
        cropped_height, cropped_width = cropped_image.shape[:2]
        logging.debug(f"Размер вырезанной части: {cropped_width}x{cropped_height}")

        # Создание нового файла изображения
        new_filename = f"{base_filename}_part_{idx}{IMAGE_FORMAT}"
        new_image_path = output_dir / new_filename

        # Сохранение вырезанного изображения
        try:
            save_image(cropped_image, new_image_path, IMAGE_FORMAT)
        except IOError as e:
            logging.error(e)
            continue  # Переход к следующей части

        # Обработка и сохранение аннотаций
        new_annotation_filename = f"{base_filename}_part_{idx}{annotation_format}"
        new_annotation_path = output_dir / new_annotation_filename

        if annotation_format == '.xml':
            # Создание нового XML-документа
            new_root = ET.Element("annotation")
            ET.SubElement(new_root, "folder").text = output_dir.name
            ET.SubElement(new_root, "filename").text = new_filename
            size = ET.SubElement(new_root, "size")
            ET.SubElement(size, "width").text = str(cropped_width)
            ET.SubElement(size, "height").text = str(cropped_height)
            ET.SubElement(size, "depth").text = str(depth)

            objects_in_part = 0

            for obj in root.findall("object"):
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    logging.warning("Найден объект без bndbox, пропуск.")
                    continue

                adjusted_coords = adjust_bndbox_voc(bndbox, split_coord, (cropped_width, cropped_height))
                if adjusted_coords is None:
                    continue  # Объект не пересекается или пересечение слишком мало

                new_xmin, new_ymin, new_xmax, new_ymax = adjusted_coords

                # Создание нового объекта в аннотациях
                new_obj = ET.SubElement(new_root, "object")
                name = obj.find("name")
                object_name = name.text if name is not None else "undefined"
                ET.SubElement(new_obj, "name").text = object_name

                new_bndbox = ET.SubElement(new_obj, "bndbox")
                ET.SubElement(new_bndbox, "xmin").text = str(new_xmin)
                ET.SubElement(new_bndbox, "ymin").text = str(new_ymin)
                ET.SubElement(new_bndbox, "xmax").text = str(new_xmax)
                ET.SubElement(new_bndbox, "ymax").text = str(new_ymax)

                objects_in_part += 1
                logging.debug(
                    f"Добавлен объект '{object_name}' с координатами: {new_xmin}, {new_ymin}, {new_xmax}, {new_ymax}")

            if objects_in_part == 0:
                logging.info(f"В части {idx} объектов нет. XML будет сохранен без элементов <object>.")

            # Сохранение нового XML-файла аннотаций
            try:
                new_tree = ET.ElementTree(new_root)
                new_tree.write(new_annotation_path, encoding='utf-8', xml_declaration=True)
                logging.debug(f"Сохранены аннотации: {new_annotation_path}")
            except Exception as e:
                logging.error(f"Ошибка при сохранении XML-аннотаций '{new_annotation_path}': {e}")

        elif annotation_format == '.txt':
            # Обработка YOLO-аннотаций
            new_yolo_annotations = []
            objects_in_part = 0

            for line in yolo_annotations:
                adjusted_coords = adjust_bndbox_yolo(line, split_coord, (cropped_width, cropped_height))
                if adjusted_coords is None:
                    continue  # Объект не пересекается или пересечение слишком мало
                new_yolo_annotations.append(adjusted_coords)
                objects_in_part += 1
                logging.debug(f"Добавлен объект из YOLO: {adjusted_coords}")

            if objects_in_part == 0:
                logging.info(f"В части {idx} объектов нет. TXT будет сохранен пустым.")

            # Сохранение нового YOLO-файла аннотаций
            try:
                with new_annotation_path.open('w') as f:
                    for ann in new_yolo_annotations:
                        f.write(f"{ann}\n")
                logging.debug(f"Сохранены аннотации YOLO: {new_annotation_path}")
            except Exception as e:
                logging.error(f"Ошибка при сохранении YOLO-аннотаций '{new_annotation_path}': {e}")

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
        split_image_and_annotations(
            image_path=image_path,
            annotation_path=annotation_path,
            output_dir=image_output_dir,
            split_h=SPLIT_HORIZONTAL,
            split_v=SPLIT_VERTICAL
        )

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
