import cv2
import os
import logging
from shapely.geometry import box
from itertools import cycle
import numpy as np

logging.basicConfig(level=logging.INFO)

def read_annotations(annotation_path):
    """
    Читает файл аннотаций и возвращает список боксов.
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
                boxes.append([int(label), float(xc), float(yc), float(w), float(h)])
            except ValueError:
                logging.warning(f"Ошибка преобразования данных в '{annotation_path}': {line}")
    return boxes


def save_image_with_bounding_boxes(img_copy, output_path):
    """
    Сохраняет изображение с русским названием.
    """
    success, encoded_image = cv2.imencode('.jpg', img_copy)
    if success:
        with open(output_path, 'wb') as f:
            f.write(encoded_image)


def draw_bounding_boxes(image_path, annotation_paths, output_dir):
    """
    Рисует bounding boxes на изображении на основе аннотаций.
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Не удалось загрузить изображение '{image_path}'")
        return

    height, width = image.shape[:2]
    colors = cycle([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)])

    for idx, ann_path in enumerate(annotation_paths):
        img_copy = image.copy()
        boxes = read_annotations(ann_path)
        color = next(colors)

        for label, xc, yc, w, h in boxes:
            x_center, y_center = xc * width, yc * height
            box_width, box_height = w * width, h * height
            x1, y1 = int(x_center - box_width / 2), int(y_center - box_height / 2)
            x2, y2 = int(x_center + box_width / 2), int(y_center + box_height / 2)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            # cv2.putText(img_copy, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        basename = os.path.basename(ann_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{basename}_bbox.jpg")
        save_image_with_bounding_boxes(img_copy, output_path)
        logging.info(f"Сохранено изображение с bounding boxes в '{output_path}'")


def calculate_iou(box1, box2):
    intersection = box1.intersection(box2).area
    union = box1.union(box2).area
    return intersection / union if union != 0 else 0


def compare_annotations(ground_truth_path, other_annotation_paths, image_path, threshold_iou=0.5, threshold_center=5, threshold_size=5):
    """
    Сравнивает главную аннотацию с другими и выводит отличия.
    """
    ground_truth_boxes = read_annotations(ground_truth_path)

    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Не удалось загрузить изображение '{image_path}'")
        return

    height, width = image.shape[:2]

    for ann_path in other_annotation_paths:
        ann_boxes = read_annotations(ann_path)
        matched_gt_indices = set()
        matched_ann_indices = set()
        iou_matrix = np.zeros((len(ground_truth_boxes), len(ann_boxes)))

        # Список для хранения различий
        box_differences = []

        # Построение матрицы IoU
        for i, b1 in enumerate(ground_truth_boxes):
            label1, xc1_norm, yc1_norm, w1_norm, h1_norm = b1
            xc1, yc1, w1, h1 = xc1_norm * width, yc1_norm * height, w1_norm * width, h1_norm * height
            x1_min, x1_max = xc1 - w1 / 2, xc1 + w1 / 2
            y1_min, y1_max = yc1 - h1 / 2, yc1 + h1 / 2
            box1 = box(x1_min, y1_min, x1_max, y1_max)

            for j, b2 in enumerate(ann_boxes):
                label2, xc2_norm, yc2_norm, w2_norm, h2_norm = b2
                xc2, yc2, w2, h2 = xc2_norm * width, yc2_norm * height, w2_norm * width, h2_norm * height
                x2_min, x2_max = xc2 - w2 / 2, xc2 + w2 / 2
                y2_min, y2_max = yc2 - h2 / 2, yc2 + h2 / 2
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
            xc1, yc1, w1, h1 = xc1_norm * width, yc1_norm * height, w1_norm * width, h1_norm * height
            xc2, yc2, w2, h2 = xc2_norm * width, yc2_norm * height, w2_norm * width, h2_norm * height

            center_diff = np.sqrt((xc1 - xc2) ** 2 + (yc1 - yc2) ** 2)
            size_diff_w = abs(w1 - w2)
            size_diff_h = abs(h1 - h2)

            significant_difference = center_diff > threshold_center or size_diff_w > threshold_size or size_diff_h > threshold_size

            box_differences.append({
                'gt_index': i,
                'ann_index': j,
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
        for diff in box_differences:
            i = diff['gt_index']
            j = diff['ann_index']
            logging.info(f"Бокс {i} из главной аннотации и бокс {j} из '{os.path.basename(ann_path)}':")
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
                logging.info(f"Бокс {idx} из главной аннотации не имеет соответствия в '{os.path.basename(ann_path)}'")
        if unmatched_ann_indices:
            for idx in unmatched_ann_indices:
                logging.info(f"Бокс {idx} из '{os.path.basename(ann_path)}' не имеет соответствия в главной аннотации")

        if not unmatched_gt_indices and not unmatched_ann_indices:
            logging.info("Все боксы из главной и сравниваемой аннотаций найдены с порогом IoU")


def main():
    image_path = r'C:\Users\mkolp\Downloads\photo_2024-10-25_09-52-57.jpg'
    ground_truth_path = r'C:\Users\mkolp\Downloads\результат.txt'
    other_annotation_paths = [r'C:\Users\mkolp\Downloads\01_1_000001.txt']
    output_dir = r'C:\Users\mkolp\Downloads\output_images'

    os.makedirs(output_dir, exist_ok=True)
    annotation_paths = [ground_truth_path] + other_annotation_paths
    draw_bounding_boxes(image_path, annotation_paths, output_dir)
    compare_annotations(ground_truth_path, other_annotation_paths, image_path, threshold_iou=0.5, threshold_center=1, threshold_size=1)


if __name__ == '__main__':
    main()