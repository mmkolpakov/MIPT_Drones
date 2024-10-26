import cv2
import os
import logging
from shapely.geometry import box
from itertools import cycle
import numpy as np

# Константы настройки
DRAW_OVERLAY = True  # Установите в True, чтобы создать изображение с наложенными разметками
OVERLAY_ANNOTATIONS = 'all'  # 'all' для всех разметок или список путей аннотаций для выборочной разметки
BOX_THICKNESS = 1  # Толщина рамок bounding boxes

# Параметры программы
IMAGE_PATH = r'C:\Users\mkolp\Downloads\photo_2024-10-25_09-52-57.jpg'

GROUND_TRUTH_PATH = r'C:\Users\mkolp\Downloads\результат.txt'
GROUND_TRUTH_IS_PIXEL_COORDINATES = False  # Установите True, если аннотации в пикселях

# Укажите, является ли каждая аннотация в пикселях (True) или нормализованной (False)
OTHER_ANNOTATION_PATHS = {
    r'C:\Users\mkolp\Downloads\Карина.txt': False,  # Нормализованные значения
    # r'C:\Users\mkolp\Downloads\другая_разметка.txt': False  # В пикселях
}

OUTPUT_DIR = r'C:\Users\mkolp\Downloads\output_images'
THRESHOLD_IOU = 0.5  # Порог IoU для соответствия боксов
THRESHOLD_CENTER = 5  # Порог для различий в центре (в пикселях)
THRESHOLD_SIZE = 5  # Порог для различий в размере (в пикселях)

# Настройка логирования
logging.basicConfig(level=logging.INFO)


def save_image_with_bounding_boxes(img_copy, output_path):
    """
    Сохраняет изображение с русским названием.
    """
    success, encoded_image = cv2.imencode('.jpg', img_copy)
    if success:
        with open(output_path, 'wb') as f:
            f.write(encoded_image)


def normalize_boxes(boxes, img_width, img_height):
    """
    Преобразует координаты из пикселей в нормализованные значения.
    """
    normalized_boxes = []
    for box in boxes:
        label, xc, yc, w, h = box
        xc_norm = xc / img_width
        yc_norm = yc / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        normalized_boxes.append([label, xc_norm, yc_norm, w_norm, h_norm])
    return normalized_boxes


def denormalize_box(box, img_width, img_height):
    """
    Преобразует координаты из нормализованных в пиксельные.
    """
    label, xc_norm, yc_norm, w_norm, h_norm = box
    xc = xc_norm * img_width
    yc = yc_norm * img_height
    w = w_norm * img_width
    h = h_norm * img_height
    return [label, xc, yc, w, h]


def read_annotations(annotation_path, img_width=None, img_height=None, pixel_coordinates=False):
    """
    Читает файл аннотаций и возвращает список боксов.
    Если pixel_coordinates=True, преобразует координаты в нормализованные.
    """
    if not os.path.exists(annotation_path):
        logging.error(f"Файл аннотаций не найден: '{annotation_path}'")
        return []

    boxes = []
    with open(annotation_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                logging.warning(f"Некорректная аннотация в '{annotation_path}': {line}")
                continue
            label, xc, yc, w, h = parts
            try:
                box = [int(label), float(xc), float(yc), float(w), float(h)]
                if pixel_coordinates:
                    if img_width and img_height:
                        # Преобразуем пиксельные координаты в нормализованные
                        box = normalize_boxes([box], img_width, img_height)[0]
                    else:
                        logging.error(
                            f"Невозможно нормализовать пиксельные координаты без указания размеров изображения.")
                        return []
                boxes.append(box)
            except ValueError:
                logging.warning(f"Ошибка преобразования данных в '{annotation_path}': {line}")
    return boxes


def draw_bounding_boxes(image_path, annotation_paths_dict, output_dir):
    """
    Рисует bounding boxes на изображении на основе аннотаций.
    Создаёт отдельные изображения для каждой аннотации.
    Дополнительно, если DRAW_OVERLAY=True, создаёт изображение с наложенными разметками всех аннотаций.
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Не удалось загрузить изображение '{image_path}'")
        return

    height, width = image.shape[:2]
    colors = cycle([
        (255, 0, 0),      # Красный
        (0, 255, 0),      # Зеленый
        (0, 0, 255),      # Синий
        (255, 255, 0),    # Желтый
        (0, 255, 255),    # Голубой
        (255, 0, 255)     # Магента
    ])

    # Создание копии для наложения разметок всех аннотаций
    if DRAW_OVERLAY:
        overlay_image = image.copy()

    for ann_path, is_pixel in annotation_paths_dict.items():
        img_copy = image.copy()
        boxes = read_annotations(ann_path, width, height, pixel_coordinates=is_pixel)
        if not boxes:
            continue
        color = next(colors)

        for box in boxes:
            label, xc_norm, yc_norm, w_norm, h_norm = box
            # Распаковываем значения, включая метку
            label, xc, yc, w_box, h_box = denormalize_box(box, width, height)
            x1 = int(xc - w_box / 2)
            y1 = int(yc - h_box / 2)
            x2 = int(xc + w_box / 2)
            y2 = int(yc + h_box / 2)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, BOX_THICKNESS)

            # Вы можете добавить отображение метки класса, если необходимо
            # cv2.putText(img_copy, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if DRAW_OVERLAY and (OVERLAY_ANNOTATIONS == 'all' or ann_path in OVERLAY_ANNOTATIONS):
                cv2.rectangle(overlay_image, (x1, y1), (x2, y2), color, BOX_THICKNESS)
                # cv2.putText(overlay_image, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        basename = os.path.basename(ann_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{basename}_bbox.jpg")
        save_image_with_bounding_boxes(img_copy, output_path)
        logging.info(f"Сохранено изображение с bounding boxes в '{output_path}'")

    if DRAW_OVERLAY:
        overlay_output_path = os.path.join(output_dir, "overlay_bbox.jpg")
        save_image_with_bounding_boxes(overlay_image, overlay_output_path)
        logging.info(f"Сохранено изображение с наложенными bounding boxes в '{overlay_output_path}'")


def calculate_iou(box1, box2):
    intersection = box1.intersection(box2).area
    union = box1.union(box2).area
    return intersection / union if union != 0 else 0


def compare_annotations(ground_truth_path, ground_truth_is_pixel, other_annotation_paths_dict, image_path,
                        threshold_iou=0.5, threshold_center=5, threshold_size=5):
    """
    Сравнивает главную аннотацию с другими и выводит отличия.
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Не удалось загрузить изображение '{image_path}'")
        return

    height, width = image.shape[:2]

    # Читаем аннотации ground truth
    ground_truth_boxes = read_annotations(ground_truth_path, img_width=width, img_height=height,
                                          pixel_coordinates=ground_truth_is_pixel)
    if not ground_truth_boxes:
        logging.error(f"Не удалось прочитать аннотации из '{ground_truth_path}'")
        return

    for ann_path, is_pixel in other_annotation_paths_dict.items():
        # Читаем другие аннотации
        ann_boxes = read_annotations(ann_path, img_width=width, img_height=height, pixel_coordinates=is_pixel)
        if not ann_boxes:
            logging.error(f"Не удалось прочитать аннотации из '{ann_path}'")
            continue

        matched_gt_indices = set()
        matched_ann_indices = set()
        iou_matrix = np.zeros((len(ground_truth_boxes), len(ann_boxes)))

        # Список для хранения различий
        box_differences = []

        # Построение матрицы IoU
        for i, b1 in enumerate(ground_truth_boxes):
            label1, xc1_norm, yc1_norm, w1_norm, h1_norm = b1
            _, xc1, yc1, w1, h1 = denormalize_box(b1, width, height)  # Правильно распаковываем значения
            x1_min, x1_max = xc1 - w1 / 2, xc1 + w1 / 2
            y1_min, y1_max = yc1 - h1 / 2, yc1 + h1 / 2
            box1 = box(x1_min, y1_min, x1_max, y1_max)

            for j, b2 in enumerate(ann_boxes):
                label2, xc2_norm, yc2_norm, w2_norm, h2_norm = b2
                _, xc2, yc2, w2, h2 = denormalize_box(b2, width, height)  # Правильно распаковываем значения
                x2_min = xc2 - w2 / 2
                x2_max = xc2 + w2 / 2
                y2_min = yc2 - h2 / 2
                y2_max = yc2 + h2 / 2
                box2 = box(x2_min, y2_min, x2_max, y2_max)

                iou = calculate_iou(box1, box2)
                iou_matrix[i, j] = iou

        # Жадный алгоритм сопоставления боксов
        while True:
            max_iou = np.max(iou_matrix)
            if max_iou < threshold_iou:
                break
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            i, j = max_idx
            matched_gt_indices.add(i)
            matched_ann_indices.add(j)

            b1 = ground_truth_boxes[i]
            b2 = ann_boxes[j]
            label1, xc1_norm, yc1_norm, w1_norm, h1_norm = b1
            label2, xc2_norm, yc2_norm, w2_norm, h2_norm = b2
            _, xc1, yc1, w1, h1 = denormalize_box(b1, width, height)[0:]
            _, xc2, yc2, w2, h2 = denormalize_box(b2, width, height)[0:]

            center_diff = np.sqrt((xc1 - xc2) ** 2 + (yc1 - yc2) ** 2)
            size_diff_w = abs(w1 - w2)
            size_diff_h = abs(h1 - h2)

            significant_difference = center_diff > threshold_center or size_diff_w > threshold_size or size_diff_h > threshold_size

            box_differences.append({
                'gt_index': i,
                'ann_index': j,
                'label_gt': label1,
                'label_ann': label2,
                'center_diff': center_diff,
                'size_diff_w': size_diff_w,
                'size_diff_h': size_diff_h,
                'iou': max_iou,
                'significant': significant_difference
            })

            # Обнуляем использованные строки и столбцы
            iou_matrix[i, :] = -1
            iou_matrix[:, j] = -1

        # Вывод подробной информации
        logging.info(f"\nОтличия между '{os.path.basename(ground_truth_path)}' и '{os.path.basename(ann_path)}':")
        if not box_differences:
            logging.info("Нет совпадающих боксов с порогом IoU.")
        for diff in box_differences:
            i = diff['gt_index']
            j = diff['ann_index']
            logging.info(f"Бокс {i} (метка {diff['label_gt']}) из главной аннотации и бокс {j} (метка {diff['label_ann']}) из '{os.path.basename(ann_path)}':")
            logging.info(f"  Смещение центров: {diff['center_diff']:.2f} пикселей")
            logging.info(f"  Разница в ширине: {diff['size_diff_w']:.2f} пикселей")
            logging.info(f"  Разница в высоте: {diff['size_diff_h']:.2f} пикселей")
            logging.info(f"  IoU: {diff['iou']:.4f}")
            if diff['significant']:
                logging.info("  → Значимые отличия по порогам")
            else:
                logging.info("  → Незначительные отличия по порогам")

        unmatched_gt_indices = set(range(len(ground_truth_boxes))) - matched_gt_indices
        unmatched_ann_indices = set(range(len(ann_boxes))) - matched_ann_indices

        if unmatched_gt_indices:
            for idx in unmatched_gt_indices:
                label = ground_truth_boxes[idx][0]
                logging.info(f"Бокс {idx} (метка {label}) из главной аннотации не имеет соответствия в '{os.path.basename(ann_path)}'")
        if unmatched_ann_indices:
            for idx in unmatched_ann_indices:
                label = ann_boxes[idx][0]
                logging.info(f"Бокс {idx} (метка {label}) из '{os.path.basename(ann_path)}' не имеет соответствия в главной аннотации")

        if not unmatched_gt_indices and not unmatched_ann_indices:
            logging.info("Все боксы из главной и сравниваемой аннотации найдены с порогом IoU")


def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        logging.error(f"Не удалось загрузить изображение '{IMAGE_PATH}'")
        return

    img_height, img_width = image.shape[:2]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Объединяем все аннотации для визуализации
    ANNOTATION_PATHS = {GROUND_TRUTH_PATH: GROUND_TRUTH_IS_PIXEL_COORDINATES}
    ANNOTATION_PATHS.update(OTHER_ANNOTATION_PATHS)

    # Рисуем bounding boxes для всех аннотаций
    draw_bounding_boxes(IMAGE_PATH, ANNOTATION_PATHS, OUTPUT_DIR)

    # Сравниваем аннотации
    compare_annotations(
        GROUND_TRUTH_PATH,
        GROUND_TRUTH_IS_PIXEL_COORDINATES,
        OTHER_ANNOTATION_PATHS,
        IMAGE_PATH,
        threshold_iou=THRESHOLD_IOU,
        threshold_center=THRESHOLD_CENTER,
        threshold_size=THRESHOLD_SIZE
    )


if __name__ == '__main__':
    main()