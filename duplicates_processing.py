import os
import re
import cv2
import logging
import albumentations as A
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional
import copy

# --------------------------- Константы и настройки ---------------------------

# Настройка логирования
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = '%(levelname)s: %(message)s'
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

# Поддерживаемые форматы изображений
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Поддерживаемые форматы аннотаций
SUPPORTED_ANNOTATION_FORMATS = ['.xml', '.txt']  # XML для PASCAL VOC, TXT для YOLO

# Словарь функций аугментации из библиотеки albumentations
AUGMENTATION_FUNCTIONS = {
    'blur': A.Blur(blur_limit=(7, 15), p=1.0),
    'horizontal_flip': A.HorizontalFlip(p=1.0),
    'color_jitter': A.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1.0
    ),
}

# Аугментации, которые применяются только один раз
DETERMINISTIC_AUGMENTATIONS = {'horizontal_flip'}

# Путь к директории с дубликатами
DUPLICATES_DIR = Path(
    r'C:\Users\mkolp\OneDrive\Изображения\test\results\duplicates'
)

# Путь к выходной директории
OUTPUT_DIR = Path(
    r'C:\Users\mkolp\OneDrive\Изображения\test\results\duplicates\output'
)

# Формат выходных изображений
IMAGE_FORMAT = '.jpg'  # Должен быть одним из SUPPORTED_IMAGE_FORMATS

# ------------------------------------------------------------------------------------


def get_class_distribution(duplicates_dir: Path, output_dir: Path) -> Dict[str, int]:
    """
    Анализирует распределение классов на основе суффиксов в именах файлов.

    :param duplicates_dir: Путь к директории с дубликатами.
    :param output_dir: Путь к выходной директории.
    :return: Словарь, сопоставляющий имена классов с их количеством.
    """
    distribution = defaultdict(int)
    for image_file in duplicates_dir.rglob('*'):
        if output_dir in image_file.parents:
            continue
        if image_file.is_file() and image_file.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
            match = re.search(r'_(\d+)\.\w+$', image_file.name)
            if match:
                class_name = f"_{match.group(1)}"
                distribution[class_name] += 1
    logging.debug(f"Распределение классов: {dict(distribution)}")
    return dict(distribution)


def load_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Загружает изображение, поддерживая пути с нелатинскими символами.

    :param image_path: Путь к изображению.
    :return: Изображение в формате NumPy массива или None, если загрузка не удалась.
    """
    try:
        with image_path.open('rb') as f:
            data = f.read()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Изображение равно None после декодирования.")
        return image
    except Exception as e:
        logging.error(f"Ошибка при загрузке изображения '{image_path}': {e}")
        return None


def save_image(image: np.ndarray, output_path: Path) -> None:
    """
    Сохраняет изображение, поддерживая пути с нелатинскими символами.

    :param image: Изображение в формате NumPy массива.
    :param output_path: Путь для сохранения изображения.
    """
    try:
        format_str = output_path.suffix.lstrip('.').lower()
        # Обработка изображений с альфа-каналом
        if image.ndim == 3 and image.shape[2] == 4 and format_str in ['jpg', 'jpeg']:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.ndim == 2 and format_str in ['jpg', 'jpeg']:
            # Преобразование градаций серого в RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        success, encoded_image = cv2.imencode(f'.{format_str}', image)
        if success:
            with output_path.open('wb') as f:
                f.write(encoded_image.tobytes())
            logging.debug(f"Изображение сохранено: {output_path}")
        else:
            raise IOError(f"Не удалось закодировать изображение для '{output_path}'.")
    except Exception as e:
        logging.error(f"Ошибка при сохранении изображения '{output_path}': {e}")
        raise


class AnnotationHandler:
    """
    Базовый класс для обработки аннотаций.
    """
    def load(self, annotation_path: Path):
        raise NotImplementedError

    def save(self, annotation, annotation_path: Path):
        raise NotImplementedError

    def update_horizontal_flip(self, annotation, image_width: int):
        raise NotImplementedError

    def update_filename(self, annotation, new_filename: str):
        raise NotImplementedError


class XMLAnnotationHandler(AnnotationHandler):
    """
    Обработчик для аннотаций в формате XML (PASCAL VOC).
    """
    def load(self, annotation_path: Path):
        try:
            tree = ET.parse(annotation_path)
            return tree
        except Exception as e:
            logging.error(f"Ошибка при загрузке XML аннотаций '{annotation_path}': {e}")
            return None

    def save(self, annotation, annotation_path: Path):
        try:
            annotation.write(annotation_path, encoding='utf-8', xml_declaration=True)
            logging.debug(f"XML аннотации сохранены: {annotation_path}")
        except Exception as e:
            logging.error(f"Ошибка при сохранении XML аннотаций '{annotation_path}': {e}")
            raise

    def update_horizontal_flip(self, annotation, image_width: int):
        """
        Обновляет координаты объектов в аннотации после горизонтального отражения.
        """
        root = annotation.getroot()
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin_node = bndbox.find('xmin')
                xmax_node = bndbox.find('xmax')
                if xmin_node is not None and xmax_node is not None:
                    try:
                        xmin = int(xmin_node.text)
                        xmax = int(xmax_node.text)
                        new_xmin = max(image_width - xmax, 0)
                        new_xmax = min(image_width - xmin, image_width)
                        xmin_node.text = str(new_xmin)
                        xmax_node.text = str(new_xmax)
                        logging.debug(f"Обновлены координаты объекта: xmin={new_xmin}, xmax={new_xmax}")
                    except ValueError as ve:
                        logging.error(f"Некорректные значения координат в XML аннотации: {ve}")
        return annotation

    def update_filename(self, annotation, new_filename: str):
        """
        Обновляет имя файла в аннотации.
        """
        root = annotation.getroot()
        filename_node = root.find('filename')
        path_node = root.find('path')
        if filename_node is not None:
            filename_node.text = new_filename
            logging.debug(f"Обновлено имя файла в аннотации: {new_filename}")
        if path_node is not None:
            path = Path(path_node.text)
            new_path = str(path.parent / new_filename)
            path_node.text = new_path
            logging.debug(f"Обновлен путь в аннотации: {new_path}")
        return annotation


class TXTAnnotationHandler(AnnotationHandler):
    """
    Обработчик для аннотаций в формате TXT (YOLO).
    """
    def load(self, annotation_path: Path):
        try:
            with annotation_path.open('r', encoding='utf-8') as f:
                lines = f.readlines()
            return lines
        except Exception as e:
            logging.error(f"Ошибка при загрузке TXT аннотаций '{annotation_path}': {e}")
            return None

    def save(self, annotation, annotation_path: Path):
        try:
            with annotation_path.open('w', encoding='utf-8') as f:
                for line in annotation:
                    f.write(line)
            logging.debug(f"TXT аннотации сохранены: {annotation_path}")
        except Exception as e:
            logging.error(f"Ошибка при сохранении TXT аннотаций '{annotation_path}': {e}")
            raise

    def update_horizontal_flip(self, annotation, image_width: int):
        """
        Обновляет координаты объектов в аннотации после горизонтального отражения.
        """
        updated_lines = []
        for line in annotation:
            parts = line.strip().split()
            if len(parts) < 5:
                logging.warning(f"Некорректная строка аннотации: {line.strip()}")
                continue
            try:
                class_id = parts[0]
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                # Отражаем x_center
                x_center = 1.0 - x_center
                x_center = max(min(x_center, 1.0), 0.0)
                updated_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                if len(parts) == 6:
                    updated_line = updated_line.strip() + f" {parts[5]}\n"
                updated_lines.append(updated_line)
                logging.debug(f"Обновлена строка аннотации: {updated_line.strip()}")
            except ValueError as ve:
                logging.error(f"Ошибка при обработке строки аннотации '{line.strip()}': {ve}")
        return updated_lines

    def update_filename(self, annotation, new_filename: str):
        # В формате YOLO имя файла не включается в файл аннотации
        return annotation


def get_annotation_handler(annotation_path: Path) -> Optional[AnnotationHandler]:
    """
    Возвращает соответствующий обработчик аннотаций на основе расширения файла.
    """
    if annotation_path.suffix.lower() == '.xml':
        return XMLAnnotationHandler()
    elif annotation_path.suffix.lower() == '.txt':
        return TXTAnnotationHandler()
    else:
        logging.error(f"Неподдерживаемый формат аннотаций: {annotation_path.suffix}")
        return None


def apply_augmentation(image: np.ndarray, augmentation_function: A.BasicTransform) -> np.ndarray:
    """
    Применяет аугментацию к изображению.

    :param image: Изображение в формате NumPy массива.
    :param augmentation_function: Функция аугментации из albumentations.
    :return: Аугментированное изображение.
    """
    augmented = augmentation_function(image=image)
    return augmented['image']


def adjust_augmentation_mapping(distribution: Dict[str, int]) -> Dict[str, str]:
    """
    Изменяет сопоставление аугментаций.

    :param distribution: Словарь распределения классов.
    :return: Словарь сопоставления классов и аугментаций.
    """
    # Сортируем классы по количеству изображений в порядке убывания
    sorted_classes = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    class_augmentation_mapping = {}

    max_count = sorted_classes[0][1]
    classes_with_max_count = [class_name for class_name, count in sorted_classes if count == max_count]

    # Приоритет распределения аугментаций
    if len(classes_with_max_count) >= 2:
        # Если два или более класса имеют максимальное количество изображений
        class_augmentation_mapping[classes_with_max_count[0]] = 'blur'
        class_augmentation_mapping[classes_with_max_count[1]] = 'horizontal_flip'
        for class_name, _ in sorted_classes:
            if class_name not in class_augmentation_mapping:
                class_augmentation_mapping[class_name] = 'color_jitter'
    else:
        # Если один класс имеет максимальное количество изображений
        class_augmentation_mapping[sorted_classes[0][0]] = 'blur'
        if len(sorted_classes) > 1:
            class_augmentation_mapping[sorted_classes[1][0]] = 'horizontal_flip'
        for class_name, _ in sorted_classes:
            if class_name not in class_augmentation_mapping:
                class_augmentation_mapping[class_name] = 'color_jitter'

    logging.debug(f"Сопоставление аугментаций: {class_augmentation_mapping}")
    return class_augmentation_mapping


def process_duplicates(duplicates_dir: Path, output_dir: Path, image_format: str) -> None:
    """
    Обрабатывает изображения и аннотации в директории дубликатов.

    :param duplicates_dir: Путь к директории с дубликатами.
    :param output_dir: Путь к выходной директории.
    :param image_format: Формат выходных изображений.
    """
    if not duplicates_dir.is_dir():
        logging.error(f"Директория дубликатов не существует: {duplicates_dir}")
        return

    distribution = get_class_distribution(duplicates_dir, output_dir)
    if not distribution:
        logging.info("Нет классов для аугментации. Завершение обработки.")
        return

    # Изменяем сопоставление аугментаций
    class_augmentation_mapping = adjust_augmentation_mapping(distribution)

    output_dir.mkdir(parents=True, exist_ok=True)

    for image_file in duplicates_dir.rglob('*'):
        if output_dir in image_file.parents:
            continue
        if not image_file.is_file():
            continue
        if image_file.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
            continue

        logging.debug(f"Обработка файла: {image_file}")

        match = re.search(r'_(\d+)\.\w+$', image_file.name)
        if not match:
            logging.warning(f"Имя файла не соответствует ожидаемому шаблону: {image_file.name}")
            continue

        class_name = f"_{match.group(1)}"

        # Получаем функцию аугментации для этого класса
        augmentation_name = class_augmentation_mapping.get(class_name, 'color_jitter')
        augmentation_function = AUGMENTATION_FUNCTIONS.get(augmentation_name)

        if augmentation_function is None:
            logging.warning(f"Не найдена функция аугментации для класса '{class_name}'. Пропуск.")
            continue

        image = load_image(image_file)
        if image is None:
            logging.error(f"Не удалось загрузить изображение: {image_file}")
            continue

        # Загружаем аннотации
        annotation_file_xml = image_file.with_suffix('.xml')
        annotation_file_txt = image_file.with_suffix('.txt')
        if annotation_file_xml.exists():
            annotation_file = annotation_file_xml
        elif annotation_file_txt.exists():
            annotation_file = annotation_file_txt
        else:
            logging.warning(f"Файл аннотаций не найден для изображения: {image_file}")
            continue

        annotation_handler = get_annotation_handler(annotation_file)
        if annotation_handler is None:
            continue

        annotation = annotation_handler.load(annotation_file)
        if annotation is None:
            logging.error(f"Не удалось загрузить аннотации для: {image_file}")
            continue

        logging.debug(f"К файлу {image_file.name} будет применена аугментация '{augmentation_name}'")

        augmented_image = apply_augmentation(image, augmentation_function)

        # Обновляем аннотации, если необходимо
        if augmentation_name == 'horizontal_flip':
            image_width = image.shape[1]
            # Создаём копию аннотаций
            augmented_annotation = annotation_handler.update_horizontal_flip(
                copy.deepcopy(annotation), image_width
            )
            logging.debug("Аннотации обновлены после горизонтального отражения")
        else:
            augmented_annotation = copy.deepcopy(annotation)  # Аннотации не изменяются

        # Подготавливаем пути для сохранения
        relative_path = image_file.relative_to(duplicates_dir)
        # Изменяем имя файла, чтобы включить название аугментации и избежать перезаписи
        output_image_name = relative_path.stem + f"_{augmentation_name}" + image_format
        output_image_path = output_dir / relative_path.parent / output_image_name
        output_annotation_path = output_image_path.with_suffix(annotation_file.suffix)

        # Убеждаемся, что директория для сохранения существует
        output_image_path.parent.mkdir(parents=True, exist_ok=True)

        # Обновляем имя файла в аннотации
        new_image_filename = output_image_path.name
        augmented_annotation = annotation_handler.update_filename(
            augmented_annotation, new_image_filename
        )

        # Сохраняем аугментированное изображение и аннотации
        try:
            save_image(augmented_image, output_image_path)
            annotation_handler.save(augmented_annotation, output_annotation_path)
            logging.info(f"Файл '{output_image_path.name}' сохранен с аугментацией '{augmentation_name}'")
        except Exception as e:
            logging.exception(f"Не удалось сохранить данные для: {image_file}, ошибка: {e}")


def main():
    """
    Основная функция скрипта.
    """
    try:
        process_duplicates(DUPLICATES_DIR, OUTPUT_DIR, IMAGE_FORMAT)
        logging.info("Обработка завершена успешно.")
    except Exception as e:
        logging.exception(f"Произошла ошибка во время обработки: {e}")


if __name__ == '__main__':
    main()
