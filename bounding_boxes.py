import cv2
import os
import logging
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ===== Константы для параметров программы =====

# Директория с изображениями и их аннотациями
IMAGE_DIR = r'C:\Users\mkolp\OneDrive\Документы\5_course\Батчи_разметка\Батчи\batch_007'

# Директория с эталонными аннотациями
GROUND_TRUTH_DIR = r'C:\Users\mkolp\Downloads\Telegram Desktop\batch_007_k'

# Формат эталонной аннотации ('txt' или 'xml')
GROUND_TRUTH_FORMAT = 'txt'  # Или 'xml', в зависимости от формата ваших аннотаций

# Формат аннотаций в директории с изображениями ('txt' или 'xml')
ANNOTATION_FORMAT = 'txt'  # Или 'xml', в зависимости от формата ваших аннотаций

# Директория для сохранения результатов
OUTPUT_DIR = r'C:\Users\mkolp\Downloads\output'

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

def read_image(image_path):
    """
    Читает изображение с поддержкой путей с нелатинскими символами.
    """
    try:
        with Image.open(image_path) as img:
            image = img.convert('RGB')
            return np.array(image)
    except Exception as e:
        logging.error(f"Не удалось загрузить изображение '{image_path}': {e}")
        return None

def save_image(img, output_path):
    """
    Сохраняет изображение с поддержкой путей с нелатинскими символами.
    """
    try:
        # Конвертируем изображение из BGR в RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(img_rgb)
        image_pil.save(output_path)
    except Exception as e:
        logging.error(f"Не удалось сохранить изображение '{output_path}': {e}")

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
    image = read_image(image_path)
    if image is None:
        logging.error(f"Не удалось загрузить изображение '{image_path}'")
        return
    # Конвертируем изображение из RGB в BGR для OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
        ann_name = os.path.basename(ann_path)
        colors[ann_name] = color
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
        image_basename = os.path.basename(image_path).split('.')[0]
        ann_basename = os.path.basename(ann_path).split('.')[0]
        output_filename = f"{image_basename}_{ann_basename}_bbox.jpg"
        output_path = os.path.join(output_dir, output_filename)
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
        overlay_output_filename = f"{os.path.basename(image_path).split('.')[0]}_overlay_bbox.jpg"
        overlay_output_path = os.path.join(output_dir, overlay_output_filename)
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

def compare_annotations(ground_truth_boxes, ann_boxes, image_shape, threshold_iou=0.5, threshold_center=5, threshold_size=5):
    """
    Сравнивает эталонную аннотацию с другой и возвращает отличия.
    """
    height, width = image_shape[:2]
    if width == 0 or height == 0:
        logging.error("Ширина или высота изображения равны нулю.")
        return [], set(), set()
    if not ground_truth_boxes or not ann_boxes:
        return [], set(), set()
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
    # Собираем информацию о несопоставленных боксах
    unmatched_gt_indices = set(range(len(ground_truth_boxes))) - matched_gt_indices
    unmatched_ann_indices = set(range(len(ann_boxes))) - matched_ann_indices
    return box_differences, unmatched_gt_indices, unmatched_ann_indices

def main():
    image_dir = IMAGE_DIR
    ground_truth_dir = GROUND_TRUTH_DIR
    output_dir = OUTPUT_DIR
    threshold_iou = THRESHOLD_IOU
    threshold_center = THRESHOLD_CENTER
    threshold_size = THRESHOLD_SIZE
    draw_overlay = DRAW_OVERLAY
    overlay_annotations = OVERLAY_ANNOTATIONS
    annotation_format = ANNOTATION_FORMAT
    ground_truth_format = GROUND_TRUTH_FORMAT
    os.makedirs(output_dir, exist_ok=True)
    # Получаем список изображений
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        logging.error(f"Не найдено изображений в директории '{image_dir}'")
        return
    # Сбор сводки по данным из эталонной и сравниваемой разметок
    summary = {}
    differences_in_counts = []
    for image_file in image_files:
        image_base_name = os.path.splitext(image_file)[0]
        ground_truth_annotation_file = image_base_name + '.' + ground_truth_format
        ground_truth_path = os.path.join(ground_truth_dir, ground_truth_annotation_file)
        annotation_file = image_base_name + '.' + annotation_format
        annotation_path = os.path.join(image_dir, annotation_file)
        image_path = os.path.join(image_dir, image_file)
        image = read_image(image_path)
        if image is None:
            continue
        image_shape = image.shape
        gt_exists = os.path.exists(ground_truth_path)
        ann_exists = os.path.exists(annotation_path)
        gt_num = None
        ann_num = None
        if gt_exists:
            ground_truth_boxes = read_annotation(ground_truth_path, img_width=image_shape[1], img_height=image_shape[0], fmt=ground_truth_format)
            gt_num = len(ground_truth_boxes)
        if ann_exists:
            ann_boxes = read_annotation(annotation_path, img_width=image_shape[1], img_height=image_shape[0], fmt=annotation_format)
            ann_num = len(ann_boxes)
        summary[image_file] = {'gt_num': gt_num, 'ann_num': ann_num, 'gt_exists': gt_exists, 'ann_exists': ann_exists}
        if gt_num is not None and ann_num is not None and gt_num != ann_num:
            differences_in_counts.append(image_file)
    # Вывод сводки
    logging.info("\nСводка по данным из эталонной разметки:")
    for image_file in image_files:
        info = summary.get(image_file, {})
        gt_exists = info.get('gt_exists', False)
        gt_num = info.get('gt_num', None)
        if not gt_exists:
            logging.info(f"Изображение '{image_file}': эталонная аннотация не найдена.")
        elif gt_num == 0:
            logging.info(f"Изображение '{image_file}': нет объектов в эталонной разметке.")
        else:
            logging.info(f"Изображение '{image_file}': {gt_num} объектов в эталонной разметке.")
    # Вывод информации о различиях в количестве объектов
    if differences_in_counts:
        logging.info("\nИзображения с различиями в количестве объектов между эталонной и сравниваемой разметками:")
        for image_file in differences_in_counts:
            info = summary[image_file]
            logging.info(f"Изображение '{image_file}': {info['gt_num']} объектов в эталонной разметке, {info['ann_num']} объектов в сравниваемой разметке.")
    else:
        logging.info("\nНет различий в количестве объектов между эталонной и сравниваемой разметками.")
    # Обработка каждого изображения
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image_base_name = os.path.splitext(image_file)[0]
        annotation_file = image_base_name + '.' + annotation_format
        annotation_path = os.path.join(image_dir, annotation_file)
        ground_truth_annotation_file = image_base_name + '.' + ground_truth_format
        ground_truth_path = os.path.join(ground_truth_dir, ground_truth_annotation_file)
        if not os.path.exists(annotation_path):
            logging.warning(f"Аннотация для изображения '{image_file}' не найдена. Пропускаем.")
            continue
        if not os.path.exists(ground_truth_path):
            logging.warning(f"Эталонная аннотация для изображения '{image_file}' не найдена. Пропускаем.")
            continue
        # Рисуем bounding boxes
        annotation_paths = [ground_truth_path, annotation_path]
        annotation_formats = [ground_truth_format, annotation_format]
        draw_bounding_boxes(image_path, annotation_paths, annotation_formats, output_dir, draw_overlay, overlay_annotations)
        # Сравниваем аннотации
        image = read_image(image_path)
        if image is None:
            continue
        image_shape = image.shape
        ground_truth_boxes = read_annotation(ground_truth_path, img_width=image_shape[1], img_height=image_shape[0], fmt=ground_truth_format)
        ann_boxes = read_annotation(annotation_path, img_width=image_shape[1], img_height=image_shape[0], fmt=annotation_format)
        differences, unmatched_gt, unmatched_ann = compare_annotations(
            ground_truth_boxes,
            ann_boxes,
            image_shape,
            threshold_iou=threshold_iou,
            threshold_center=threshold_center,
            threshold_size=threshold_size
        )
        # Выводим результаты сравнения
        logging.info(f"\nСравнение аннотаций для изображения '{image_file}':")
        if differences:
            for diff in differences:
                logging.info(f"Бокс {diff['gt_index']} (метка '{diff['label_gt']}') из эталонной аннотации и бокс {diff['ann_index']} (метка '{diff['label_ann']}') из аннотации изображения:")
                logging.info(f"  Смещение центров: {diff['center_diff']:.2f} пикселей")
                logging.info(f"  Разница в ширине: {diff['size_diff_w']:.2f} пикселей")
                logging.info(f"  Разница в высоте: {diff['size_diff_h']:.2f} пикселей")
                logging.info(f"  IoU: {diff['iou']:.4f}")
                if diff['significant']:
                    logging.info("  → Значимые отличия по порогам")
                else:
                    logging.info("  → Незначительные отличия по порогам")
        else:
            logging.info("Нет совпадающих боксов с порогом IoU.")
        if unmatched_gt:
            for idx in unmatched_gt:
                label = ground_truth_boxes[idx][0]
                logging.info(f"Бокс {idx} (метка '{label}') из эталонной аннотации не имеет соответствия в аннотации изображения")
        if unmatched_ann:
            for idx in unmatched_ann:
                label = ann_boxes[idx][0]
                logging.info(f"Бокс {idx} (метка '{label}') из аннотации изображения не имеет соответствия в эталонной аннотации")
        if not differences and not unmatched_gt and not unmatched_ann:
            logging.info("Все боксы из эталонной и сравниваемой аннотаций найдены с порогом IoU")

if __name__ == '__main__':
    main()