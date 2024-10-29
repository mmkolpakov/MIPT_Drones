import cv2
import os
import logging
import numpy as np
import xml.etree.ElementTree as ET

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ===== Константы для параметров программы =====

# Путь к изображению
IMAGE_PATH = r'C:\Users\mkolp\Downloads\DJI_020240907173008.jpg'

# Путь к файлу с эталонными аннотациями
GROUND_TRUTH_PATH = r'C:\Users\mkolp\Downloads\DJI_020240907173008.xml'

# Формат эталонной аннотации ('txt' или 'xml')
GROUND_TRUTH_FORMAT = 'xml'

# Список путей к другим файлам аннотаций
OTHER_ANNOTATION_PATHS = [
    r'C:\Users\mkolp\Downloads\Карина.xml',
    # Добавьте другие пути к файлам аннотаций при необходимости
]

# Список форматов аннотаций для других файлов ('txt' или 'xml')
OTHER_ANNOTATION_FORMATS = [
    'xml',  # Формат для 'Карина.xml'
    # Укажите форматы для других аннотаций в том же порядке, что и пути
]

# Директория для сохранения результатов
OUTPUT_DIR = r'C:\Users\mkolp\Downloads\test.xml'

# Порог IoU для соответствия боксов
THRESHOLD_IOU = 0.5

# Порог для различий в центре (в пикселях)
THRESHOLD_CENTER = 5

# Порог для различий в размере (в пикселях)
THRESHOLD_SIZE = 5

# Флаг создания изображения с наложенными разметками
DRAW_OVERLAY = True

# Список аннотаций для наложения или 'all' для всех
OVERLAY_ANNOTATIONS = 'all'

# ===== Функции программы =====

def save_image(img, output_path):
    """
    Сохраняет изображение с поддержкой путей с нелатинскими символами.
    """
    success, encoded_image = cv2.imencode('.jpg', img)
    if success:
        with open(output_path, 'wb') as f:
            f.write(encoded_image)
    else:
        logging.error(f"Не удалось сохранить изображение '{output_path}'")

def normalize_box(box, img_width, img_height):
    """
    Преобразует координаты из пикселей в нормализованные значения.
    """
    label, xc, yc, w, h = box
    if img_width == 0 or img_height == 0:
        raise ValueError("Ширина или высота изображения равны нулю.")
    xc_norm = xc / img_width
    yc_norm = yc / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return [label, xc_norm, yc_norm, w_norm, h_norm]

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

def read_txt_annotation(annotation_path, img_width, img_height):
    """
    Читает TXT-аннотацию и возвращает список нормализованных боксов.
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
                box = [label.strip(), float(xc), float(yc), float(w), float(h)]
                boxes.append(box)
            except ValueError:
                logging.warning(f"Ошибка преобразования данных в '{annotation_path}': {line}")
    return boxes

def read_xml_annotation(annotation_path, img_width, img_height):
    """
    Читает XML-аннотацию и возвращает список нормализованных боксов.
    """
    if not os.path.exists(annotation_path):
        logging.error(f"Файл аннотаций не найден: '{annotation_path}'")
        return []
    boxes = []
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            try:
                label = obj.find('name').text.strip()
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                xc = (xmin + xmax) / 2
                yc = (ymin + ymax) / 2
                w = xmax - xmin
                h = ymax - ymin
                if img_width == 0 or img_height == 0:
                    logging.error("Ширина или высота изображения равны нулю.")
                    return []
                box = [label, xc / img_width, yc / img_height, w / img_width, h / img_height]
                boxes.append(box)
            except Exception as e:
                logging.warning(f"Ошибка в объекте аннотации в '{annotation_path}': {e}")
    except ET.ParseError as e:
        logging.error(f"Ошибка парсинга XML-файла '{annotation_path}': {e}")
    return boxes

def read_annotation(annotation_path, img_width, img_height, fmt='txt'):
    """
    Унифицированная функция для чтения аннотаций разных форматов.
    """
    if fmt == 'txt':
        return read_txt_annotation(annotation_path, img_width, img_height)
    elif fmt == 'xml':
        return read_xml_annotation(annotation_path, img_width, img_height)
    else:
        logging.error(f"Неподдерживаемый формат аннотаций '{fmt}' для файла '{annotation_path}'")
        return []

def draw_bounding_boxes(image_path, annotation_paths, annotation_formats, output_dir, draw_overlay=False, overlay_annotations='all'):
    """
    Рисует bounding boxes на изображении на основе аннотаций.
    Создает отдельные изображения для каждой аннотации.
    Если draw_overlay=True, создает изображение с наложенными разметками всех аннотаций.
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Не удалось загрузить изображение '{image_path}'")
        return
    height, width = image.shape[:2]
    if width == 0 or height == 0:
        logging.error("Ширина или высота изображения равны нулю.")
        return
    colors = {}
    color_palette = [
        (255, 0, 0),      # Красный
        (0, 255, 0),      # Зеленый
        (0, 0, 255),      # Синий
        (255, 255, 0),    # Желтый
        (0, 255, 255),    # Голубой
        (255, 0, 255)     # Магента
    ]
    # Создание копии для наложения разметок всех аннотаций
    if draw_overlay:
        overlay_image = image.copy()
    for idx, ann_path in enumerate(annotation_paths):
        img_copy = image.copy()
        fmt = annotation_formats[idx] if annotation_formats else 'txt'
        boxes = read_annotation(ann_path, width, height, fmt)
        if not boxes:
            continue
        color = color_palette[idx % len(color_palette)]
        colors[os.path.basename(ann_path)] = color
        for box in boxes:
            label, xc_norm, yc_norm, w_norm, h_norm = box
            label, xc, yc, w_box, h_box = denormalize_box(box, width, height)
            x1 = int(xc - w_box / 2)
            y1 = int(yc - h_box / 2)
            x2 = int(xc + w_box / 2)
            y2 = int(yc + h_box / 2)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_copy, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            if draw_overlay and (overlay_annotations == 'all' or ann_path in overlay_annotations):
                cv2.rectangle(overlay_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(overlay_image, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        basename = os.path.basename(ann_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{basename}_bbox.jpg")
        save_image(img_copy, output_path)
        logging.info(f"Сохранено изображение с bounding boxes в '{output_path}'")
    if draw_overlay:
        # Добавляем легенду
        legend_height = 25 * len(colors)
        channels = overlay_image.shape[2] if len(overlay_image.shape) > 2 else 1
        overlay_image_with_legend = np.zeros((height + legend_height, width, channels), dtype=np.uint8)
        overlay_image_with_legend[:height] = overlay_image
        for idx, (name, color) in enumerate(colors.items()):
            cv2.rectangle(overlay_image_with_legend, (10, height + idx * 25 + 5), (30, height + idx * 25 + 20), color, -1)
            cv2.putText(overlay_image_with_legend, name, (40, height + idx * 25 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        overlay_output_path = os.path.join(output_dir, "overlay_bbox.jpg")
        save_image(overlay_image_with_legend, overlay_output_path)
        logging.info(f"Сохранено изображение с наложенными bounding boxes в '{overlay_output_path}'")

def calculate_iou(box1_coords, box2_coords):
    """
    Вычисляет IoU между двумя боксами.
    """
    x_left = max(box1_coords[0], box2_coords[0])
    y_top = max(box1_coords[1], box2_coords[1])
    x_right = min(box1_coords[2], box2_coords[2])
    y_bottom = min(box1_coords[3], box2_coords[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1_coords[2] - box1_coords[0]) * (box1_coords[3] - box1_coords[1])
    box2_area = (box2_coords[2] - box2_coords[0]) * (box2_coords[3] - box2_coords[1])
    if box1_area + box2_area - intersection_area == 0:
        return 0.0
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def compare_annotations(ground_truth_path, ground_truth_format, other_annotation_paths, annotation_formats,
                        image_path, threshold_iou=0.5, threshold_center=5, threshold_size=5):
    """
    Сравнивает главную аннотацию с другими и выводит отличия.
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Не удалось загрузить изображение '{image_path}'")
        return
    height, width = image.shape[:2]
    if width == 0 or height == 0:
        logging.error("Ширина или высота изображения равны нулю.")
        return
    # Читаем аннотации ground truth
    ground_truth_boxes = read_annotation(ground_truth_path, img_width=width, img_height=height, fmt=ground_truth_format)
    if not ground_truth_boxes:
        logging.error(f"Не удалось прочитать аннотации из '{ground_truth_path}'")
        return
    for idx, ann_path in enumerate(other_annotation_paths):
        fmt = annotation_formats[idx] if annotation_formats else 'txt'
        ann_boxes = read_annotation(ann_path, img_width=width, img_height=height, fmt=fmt)
        if not ann_boxes:
            logging.warning(f"Не удалось прочитать аннотации из '{ann_path}'")
            continue
        if len(ground_truth_boxes) == 0 or len(ann_boxes) == 0:
            logging.info("Нет боксов для сравнения.")
            continue
        matched_gt_indices = set()
        matched_ann_indices = set()
        iou_matrix = np.zeros((len(ground_truth_boxes), len(ann_boxes)))
        # Построение матрицы IoU
        for i, b1 in enumerate(ground_truth_boxes):
            label1, xc1_norm, yc1_norm, w1_norm, h1_norm = b1
            _, xc1, yc1, w1, h1 = denormalize_box(b1, width, height)
            x1_min, x1_max = xc1 - w1 / 2, xc1 + w1 / 2
            y1_min, y1_max = yc1 - h1 / 2, yc1 + h1 / 2
            for j, b2 in enumerate(ann_boxes):
                label2, xc2_norm, yc2_norm, w2_norm, h2_norm = b2
                _, xc2, yc2, w2, h2 = denormalize_box(b2, width, height)
                x2_min, x2_max = xc2 - w2 / 2, xc2 + w2 / 2
                y2_min, y2_max = yc2 - h2 / 2, yc2 + h2 / 2
                iou = calculate_iou([x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max])
                iou_matrix[i, j] = iou
        # Жадный алгоритм сопоставления боксов
        box_differences = []
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
            _, xc1, yc1, w1, h1 = denormalize_box(b1, width, height)
            _, xc2, yc2, w2, h2 = denormalize_box(b2, width, height)
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
            iou_matrix[i, :] = -1
            iou_matrix[:, j] = -1
        # Вывод подробной информации
        logging.info(f"\nОтличия между '{os.path.basename(ground_truth_path)}' и '{os.path.basename(ann_path)}':")
        if not box_differences:
            logging.info("Нет совпадающих боксов с порогом IoU.")
        for diff in box_differences:
            i = diff['gt_index']
            j = diff['ann_index']
            logging.info(f"Бокс {i} (метка '{diff['label_gt']}') из главной аннотации и бокс {j} (метка '{diff['label_ann']}') из '{os.path.basename(ann_path)}':")
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
                logging.info(f"Бокс {idx} (метка '{label}') из главной аннотации не имеет соответствия в '{os.path.basename(ann_path)}'")
        if unmatched_ann_indices:
            for idx in unmatched_ann_indices:
                label = ann_boxes[idx][0]
                logging.info(f"Бокс {idx} (метка '{label}') из '{os.path.basename(ann_path)}' не имеет соответствия в главной аннотации")
        if not unmatched_gt_indices and not unmatched_ann_indices:
            logging.info("Все боксы из главной и сравниваемой аннотации найдены с порогом IoU")

def main():
    image_path = IMAGE_PATH
    ground_truth_path = GROUND_TRUTH_PATH
    other_annotation_paths = OTHER_ANNOTATION_PATHS
    output_dir = OUTPUT_DIR
    threshold_iou = THRESHOLD_IOU
    threshold_center = THRESHOLD_CENTER
    threshold_size = THRESHOLD_SIZE
    draw_overlay = DRAW_OVERLAY
    overlay_annotations = OVERLAY_ANNOTATIONS
    annotation_formats = OTHER_ANNOTATION_FORMATS
    ground_truth_format = GROUND_TRUTH_FORMAT
    os.makedirs(output_dir, exist_ok=True)
    # Проверка наличия файлов
    if not os.path.exists(image_path):
        logging.error(f"Изображение '{image_path}' не найдено.")
        return
    if not os.path.exists(ground_truth_path):
        logging.error(f"Файл с эталонными аннотациями '{ground_truth_path}' не найден.")
        return
    if len(other_annotation_paths) != len(annotation_formats):
        logging.error("Количество путей аннотаций и форматов не совпадает.")
        return
    for ann_path in other_annotation_paths:
        if not os.path.exists(ann_path):
            logging.warning(f"Файл аннотаций '{ann_path}' не найден. Пропускаем.")
            continue
    # Рисуем bounding boxes для всех аннотаций
    all_annotation_paths = [ground_truth_path] + other_annotation_paths
    all_annotation_formats = [ground_truth_format] + annotation_formats
    draw_bounding_boxes(image_path, all_annotation_paths, all_annotation_formats, output_dir, draw_overlay, overlay_annotations)
    # Сравниваем аннотации
    compare_annotations(
        ground_truth_path,
        ground_truth_format,
        other_annotation_paths,
        annotation_formats,
        image_path,
        threshold_iou=threshold_iou,
        threshold_center=threshold_center,
        threshold_size=threshold_size
    )

if __name__ == '__main__':
    main()