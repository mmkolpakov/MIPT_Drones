import os
import random
import exifread
import math
import csv
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from shapely.geometry import box
from shapely.strtree import STRtree
from pyproj import Transformer
from dateutil import parser
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from geopy.distance import geodesic
from shapely.ops import unary_union
from typing import List, Dict, Any, Tuple
import folium
import logging
import time
import itertools
import json
from rtree import index

# Настраиваемые параметры
ROOT_DIR = r'C:\Users\mkolp\OneDrive\Документы\Данные для разметки\Датасет'  # Корневой каталог с изображениями
OUTPUT_CSV = 'output.csv'  # Имя выходного CSV файла
OUTPUT_EXCEL = 'output.xlsx'  # Имя выходного Excel файла
SENSOR_WIDTH_MM = 36.0  # Ширина сенсора в мм
SENSOR_HEIGHT_MM = 24.0  # Высота сенсора в мм
OBJECT_SIZE_M = 1.0  # Размер объекта в метрах для расчета size_in_pixels
MAX_WORKERS = 4  # Максимальное количество потоков для параллельной обработки
MAX_FILES_TO_PROCESS = 10000  # Максимальное количество файлов для обработки

# Параметры для выбора и копирования изображений
NUM_SETS_TO_CREATE = 2  # Количество сетов (папок) для создания
NUM_IMAGES_PER_SET = 50  # Количество изображений в каждом сете
OVERLAP_PERCENTAGE_BETWEEN_SETS = 20.0  # Процент одинаковых изображений между сетами
EXCLUDE_KEYWORD = 'дети'  # Ключевое слово для исключения изображений
DESTINATION_DIR = r'C:\Users\mkolp\OneDrive\Документы\Selected_Images'  # Папка назначения для копирования

# Параметры высоты для отбора изображений
ALTITUDE_RANGES = [(0, 200)]  # Диапазоны высот в метрах

# Флаги для управления визуализацией
VISUALIZE_ALL_IMAGES = True  # Если True, визуализируются все изображения
VISUALIZE_SELECTED_IMAGES = True  # Если True, визуализируются отобранные изображения

DEBUG = True  # Флаг отладки. Установите в True для вывода подробной информации

CACHE_FILE = 'elevation_cache.json'  # Имя JSON-файла для кэширования высот

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

# Создаем обработчик для записи логов в файл при DEBUG
if DEBUG:
    log_filename = 'debug.log'
    fh = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# Создаем обработчик для вывода логов в консоль
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)  # Выводим INFO и выше в консоль
formatter_console = logging.Formatter('%(message)s')
ch.setFormatter(formatter_console)
logger.addHandler(ch)


def get_decimal_from_dms(dms, ref):
    """Преобразует координаты GPS из формата DMS в десятичные градусы."""
    degrees = dms.values[0].num / dms.values[0].den
    minutes = dms.values[1].num / dms.values[1].den
    seconds = dms.values[2].num / dms.values[2].den

    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal


def parse_gps_and_altitude(tags):
    """Извлекает GPS информацию и высоту из EXIF тегов."""
    latitude = longitude = altitude = None
    gps_tags = {
        'GPS GPSLatitude': 'latitude',
        'GPS GPSLatitudeRef': 'lat_ref',
        'GPS GPSLongitude': 'longitude',
        'GPS GPSLongitudeRef': 'lon_ref',
    }

    gps_data = {}
    for tag, key in gps_tags.items():
        if tag in tags:
            gps_data[key] = tags[tag]

    if all(k in gps_data for k in ('latitude', 'lat_ref', 'longitude', 'lon_ref')):
        lat = gps_data['latitude']
        lat_ref = gps_data['lat_ref'].values
        lon = gps_data['longitude']
        lon_ref = gps_data['lon_ref'].values

        latitude = get_decimal_from_dms(lat, lat_ref)
        longitude = get_decimal_from_dms(lon, lon_ref)

    # Используем GPSAltitude (абсолютная высота)
    if 'GPS GPSAltitude' in tags:
        alt = tags['GPS GPSAltitude']
        altitude = alt.values[0].num / alt.values[0].den
        # Проверяем AltitudeRef
        alt_ref = tags.get('GPS GPSAltitudeRef')
        if alt_ref and alt_ref.values[0] == 1:
            altitude = -altitude
    else:
        altitude = None

    return latitude, longitude, altitude


def calculate_field_of_view(sensor_size_mm, focal_length_mm):
    """Вычисляет угол обзора в градусах."""
    fov = 2 * math.degrees(math.atan(sensor_size_mm / (2 * focal_length_mm)))
    logger.debug(f"FOV (sensor_size_mm={sensor_size_mm}, focal_length_mm={focal_length_mm}): {fov} degrees")
    return fov


def compute_file_hash(file_path):
    """Вычисляет хеш файла для обнаружения дубликатов."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as afile:
        for chunk in iter(lambda: afile.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def process_file(file_info, transformer, ref_x, ref_y, hash_set, utm_epsg_code):
    """Обрабатывает один файл изображения."""
    file_path, relative_path, filename = file_info
    data = {}
    try:
        # Вычисляем хеш файла для обнаружения дубликатов
        file_hash = compute_file_hash(file_path)
        if file_hash in hash_set:
            logger.warning(f"Предупреждение: Обнаружен дубликат файла {filename}. Запись будет пропущена.")
            return None  # Пропускаем обработку этого файла
        else:
            hash_set.add(file_hash)

        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)

            if not tags:
                logger.warning(f"Предупреждение: EXIF-данные не найдены в файле {filename}.")
                return None

            # Извлекаем данные GPS и высоту
            latitude, longitude, altitude = parse_gps_and_altitude(tags)

            # Отладочный вывод
            logger.debug(f"[{filename}] latitude: {latitude}, longitude: {longitude}, altitude: {altitude}")

            # Устанавливаем terrain_elevation и relative_altitude в None для последующей обработки
            terrain_elevation = None
            relative_altitude = None

            # Извлекаем время съемки
            datetime_original = tags.get('EXIF DateTimeOriginal', tags.get('Image DateTime'))
            if datetime_original:
                dt_str = datetime_original.values
                try:
                    shooting_time = parser.parse(dt_str)
                except (ValueError, TypeError):
                    shooting_time = None
            else:
                shooting_time = None

            # Извлекаем размеры изображения
            image_width = tags.get('EXIF ExifImageWidth', tags.get('Image ImageWidth'))
            image_height = tags.get('EXIF ExifImageLength', tags.get('Image ImageLength'))
            if image_width:
                image_width = int(str(image_width))
            else:
                image_width = None
            if image_height:
                image_height = int(str(image_height))
            else:
                image_height = None

            # Извлекаем фокусное расстояние
            focal_length_tag = tags.get('EXIF FocalLength')
            if focal_length_tag:
                focal_length = focal_length_tag.values[0]
                focal_length = focal_length.num / focal_length.den
            else:
                focal_length = None

            # Извлекаем DigitalZoomRatio
            digital_zoom_ratio_tag = tags.get('EXIF DigitalZoomRatio') or tags.get('DigitalZoomRatio')
            if digital_zoom_ratio_tag:
                try:
                    digital_zoom_ratio = float(str(digital_zoom_ratio_tag.values))
                except (ValueError, TypeError):
                    digital_zoom_ratio = 1.0  # По умолчанию 1.0, если не удалось преобразовать
            else:
                digital_zoom_ratio = 1.0

            # Корректируем фокусное расстояние с учетом DigitalZoomRatio
            if focal_length:
                effective_focal_length = focal_length * digital_zoom_ratio
            else:
                effective_focal_length = None

            data = {
                'filename': filename,
                'relative_path': relative_path,
                'file_path': file_path,
                'latitude': latitude,
                'longitude': longitude,
                'altitude': altitude,
                'terrain_elevation': terrain_elevation,  # Будет заполнено позже
                'relative_altitude': relative_altitude,  # Будет заполнено позже
                'shooting_time': shooting_time,
                'image_width': image_width,
                'image_height': image_height,
                'focal_length': focal_length,
                'digital_zoom_ratio': digital_zoom_ratio,
                'effective_focal_length': effective_focal_length,
                'utm_epsg_code': utm_epsg_code,
                'ref_x': ref_x,
                'ref_y': ref_y,
            }

            # Преобразование координат в метры относительно референсной точки
            if None not in (latitude, longitude):
                x, y = transformer.transform(longitude, latitude)
                data['x'] = x - ref_x
                data['y'] = y - ref_y

                # Отладочный вывод
                logger.debug(f"[{filename}] x: {data['x']}, y: {data['y']}")
            else:
                data['x'] = data['y'] = None
                logger.warning(f"[{filename}] Координаты x или y не определены.")

            return data
    except Exception as e:
        logger.error(f"Ошибка при обработке файла {filename}: {e}")
        return None


def load_cache():
    """Загружает кэш из файла."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache):
    """Сохраняет кэш в файл."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)


def hash_coords(coords):
    """Создает хэш для уникальной идентификации набора координат."""
    # Сортируем координаты по значениям 'latitude' и 'longitude'
    sorted_coords = sorted(coords, key=lambda x: (x['latitude'], x['longitude']))
    coords_str = json.dumps(sorted_coords, ensure_ascii=False)
    return hashlib.md5(coords_str.encode('utf-8')).hexdigest()


def fetch_elevations(data_list):
    """Получает высоты местности для всех точек и обновляет data_list с использованием кэша."""
    # Загружаем кэш
    cache = load_cache()

    # Собираем уникальные координаты
    coords = []
    data_with_coords = []
    for data in data_list:
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        if latitude is not None and longitude is not None:
            coords.append({'latitude': latitude, 'longitude': longitude})
            data_with_coords.append(data)

    if not coords:
        logger.info("Нет координат для получения высот местности.")
        return

    # Создаем хэш для текущего набора координат
    coords_hash = hash_coords(coords)

    # Проверяем, есть ли результаты в кэше
    if coords_hash in cache:
        logger.info("Используем кэшированные данные высот.")
        results = cache[coords_hash]
        # Обновляем data_list с кэшированными высотами
        for idx, result in enumerate(results):
            elevation = result.get('elevation')
            data_with_coords[idx]['terrain_elevation'] = elevation
            altitude = data_with_coords[idx].get('altitude')
            if elevation is not None and altitude is not None:
                data_with_coords[idx]['relative_altitude'] = altitude - elevation
            else:
                data_with_coords[idx]['relative_altitude'] = None
        return
    else:
        logger.info("Высоты не найдены в кэше, выполняем запрос к API...")

    # Разбиваем на батчи по 100 координат (ограничение API)
    batch_size = 100
    all_results = []
    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i + batch_size]
        batch_data = data_with_coords[i:i + batch_size]

        # Отправляем POST запрос к API
        logger.info(f"Получение высот местности для точек {i} - {i + len(batch_coords)}...")
        success = False
        retries = 0
        max_retries = 5
        backoff_factor = 1  # seconds

        while not success and retries < max_retries:
            try:
                url = 'https://api.open-elevation.com/api/v1/lookup'
                headers = {'Content-Type': 'application/json'}
                data = {'locations': batch_coords}
                response = requests.post(url, json=data, headers=headers, timeout=10)
                if response.status_code == 200:
                    results = response.json()['results']
                    if len(results) != len(batch_data):
                        logger.warning("Количество полученных высот не соответствует количеству точек в батче.")
                        break  # Не можем продолжить с этим батчем
                    # Обновляем data_list с высотами и сохраняем результаты для кэширования
                    for idx, result in enumerate(results):
                        elevation = result.get('elevation')
                        batch_data[idx]['terrain_elevation'] = elevation
                        altitude = batch_data[idx].get('altitude')
                        if elevation is not None and altitude is not None:
                            batch_data[idx]['relative_altitude'] = altitude - elevation
                            logger.debug(f"[{batch_data[idx]['filename']}] Altitude: {altitude}, Terrain Elevation: {elevation}, Relative Altitude: {batch_data[idx]['relative_altitude']}")
                        else:
                            batch_data[idx]['relative_altitude'] = None
                    all_results.extend(results)
                    success = True
                else:
                    logger.warning(f"Ошибка запроса к API open-elevation: {response.status_code}")
                    if response.status_code in [500, 502, 503, 504]:
                        # Server error, retry after delay
                        retries += 1
                        sleep_time = backoff_factor * (2 ** (retries - 1))
                        logger.info(f"Ждем {sleep_time} секунд перед повторной попыткой...")
                        time.sleep(sleep_time)
                    else:
                        # Other error, don't retry
                        break
            except requests.exceptions.RequestException as e:
                logger.error(f"Ошибка при выполнении запроса: {e}")
                retries += 1
                sleep_time = backoff_factor * (2 ** (retries - 1))
                logger.info(f"Ждем {sleep_time} секунд перед повторной попыткой...")
                time.sleep(sleep_time)
        if not success:
            logger.error(f"Не удалось получить высоты местности для точек {i} - {i + len(batch_coords)} после {max_retries} попыток.")
        else:
            # Задержка между успешными запросами, чтобы не перегружать API
            time.sleep(1)

    # Сохраняем результаты в кэш
    cache[coords_hash] = all_results
    save_cache(cache)

    logger.info("Высоты успешно получены и кэшированы.")


def calculate_additional_parameters(data_list):
    """Вычисляет дополнительные параметры для каждого изображения."""
    previous_data = None
    for data in data_list:
        effective_focal_length = data.get('effective_focal_length')
        relative_altitude = data.get('relative_altitude')
        image_width = data.get('image_width')
        image_height = data.get('image_height')
        latitude = data.get('latitude')
        longitude = data.get('longitude')

        # Проверяем наличие необходимых данных
        if None in (effective_focal_length, relative_altitude, image_width, image_height):
            logger.warning(f"Предупреждение: Недостаточно данных для вычисления параметров изображения {data['filename']}.")
            data['width_meters'] = data['height_meters'] = data['resolution_cm_per_pixel'] = data['size_in_pixels'] = None
            continue

        # Углы обзора
        data['fov_horizontal'] = calculate_field_of_view(SENSOR_WIDTH_MM, effective_focal_length)
        data['fov_vertical'] = calculate_field_of_view(SENSOR_HEIGHT_MM, effective_focal_length)

        # Размеры на земле и разрешение
        # Приводим всё к метрам
        sensor_width_m = SENSOR_WIDTH_MM / 1000  # мм в метры
        sensor_height_m = SENSOR_HEIGHT_MM / 1000
        focal_length_m = effective_focal_length / 1000
        altitude_m = relative_altitude  # Используем относительную высоту

        # Проверяем, чтобы фокусное расстояние не было нулевым
        if focal_length_m == 0:
            logger.warning(f"Предупреждение: Фокусное расстояние равно нулю для изображения {data['filename']}.")
            data['width_meters'] = data['height_meters'] = data['resolution_cm_per_pixel'] = data['size_in_pixels'] = None
            continue

        gsd = altitude_m / focal_length_m
        width_meters = sensor_width_m * gsd
        height_meters = sensor_height_m * gsd
        resolution_cm_per_pixel = (altitude_m * sensor_width_m) / (focal_length_m * image_width) * 100  # Метры в сантиметры
        size_in_pixels = (OBJECT_SIZE_M * 100) / resolution_cm_per_pixel  # Для объекта размером 1 метр

        data['width_meters'] = round(width_meters, 3)
        data['height_meters'] = round(height_meters, 3)
        data['resolution_cm_per_pixel'] = round(resolution_cm_per_pixel, 3)
        data['size_in_pixels'] = round(size_in_pixels, 2)

        logger.debug(f"[{data['filename']}] GSD: {gsd}, Width on ground: {width_meters}, Height on ground: {height_meters}")
        logger.debug(f"[{data['filename']}] Resolution (cm/pixel): {resolution_cm_per_pixel}, Size in pixels: {size_in_pixels}")

        # Расстояние до предыдущего снимка, используя геодезическое расстояние
        if previous_data and None not in (latitude, longitude, previous_data['latitude'], previous_data['longitude']):
            coords_1 = (latitude, longitude)
            coords_2 = (previous_data['latitude'], previous_data['longitude'])
            distance = geodesic(coords_1, coords_2).meters
            data['distance_to_previous'] = round(distance, 2)
            logger.debug(f"[{data['filename']}] Distance to previous image: {distance} meters")
        else:
            data['distance_to_previous'] = None

        previous_data = data

def calculate_overlap(data_list: List[Dict[str, Any]]) -> None:
    """
    Вычисляет перекрытия между изображениями, гарантируя отсутствие многократного счета.
    """
    logger.info("Создание полигонов для каждого изображения...")
    data_with_polygons = []
    geometries = []

    # Создаем полигоны для изображений с валидными координатами и размерами
    for data in data_list:
        filename = data.get('filename', 'Unknown')

        x = data.get('x')
        y = data.get('y')
        width = data.get('width_meters')
        height = data.get('height_meters')

        if None in (x, y, width, height):
            logger.warning(
                f"[{filename}] Отсутствуют необходимые параметры для создания полигона. Установка значений по умолчанию."
            )
            data['polygon'] = None
            data['area'] = 0
            data['overlap_area'] = 0.0
            data['overlap_percentage'] = 0.0
            continue

        try:
            x = float(x)
            y = float(y)
            width = float(width)
            height = float(height)
        except (TypeError, ValueError) as e:
            logger.warning(f"[{filename}] Некорректные данные: {e}. Установка значений по умолчанию.")
            data['polygon'] = None
            data['area'] = 0
            data['overlap_area'] = 0.0
            data['overlap_percentage'] = 0.0
            continue

        half_width = width / 2
        half_height = height / 2

        # Создаем полигон
        polygon = box(
            x - half_width,
            y - half_height,
            x + half_width,
            y + half_height
        )

        data['polygon'] = polygon
        data['area'] = polygon.area
        data['overlap_area'] = 0.0
        data['overlap_percentage'] = 0.0

        logger.debug(f"[{filename}] Polygon area: {data['area']}")

        data_with_polygons.append(data)
        geometries.append(polygon)

    if not data_with_polygons:
        logger.info("Нет полигонов для расчета перекрытия.")
        return

    # Создаем STRtree для поиска пересечений
    logger.info("Создание STRtree для пространственного индекса...")
    tree = STRtree(geometries)

    # Вычисляем площадь перекрытия для каждого изображения
    logger.info("Вычисление площади перекрытия для каждого изображения...")
    for i, geom in enumerate(geometries):
        data = data_with_polygons[i]
        filename = data.get('filename', 'Unknown')

        # Находим геометрии, которые потенциально пересекаются с текущей
        possible_matches_idx = tree.query(geom)

        # Исключаем саму себя
        possible_matches_idx = [idx for idx in possible_matches_idx if idx != i]

        if not possible_matches_idx:
            data['overlap_area'] = 0.0
            continue

        # Вычисляем объединение всех пересечений
        overlapping_areas = []
        for idx in possible_matches_idx:
            other_geom = geometries[idx]
            intersection = geom.intersection(other_geom)
            if not intersection.is_empty:
                overlapping_areas.append(intersection)

        if overlapping_areas:
            # Объединяем все пересечения, чтобы избежать многократного счета
            overlap_union = unary_union(overlapping_areas)
            total_overlap_area = overlap_union.area
        else:
            total_overlap_area = 0.0

        data['overlap_area'] = total_overlap_area
        logger.debug(f"[{filename}] Total overlap area: {total_overlap_area}")

    # Вычисляем процент перекрытия для каждого изображения
    logger.info("Вычисление процента перекрытия...")
    for data in data_with_polygons:
        area = data.get('area', 0)
        overlap_area = data.get('overlap_area', 0.0)
        if area > 0:
            overlap_percentage = (overlap_area / area) * 100
            data['overlap_percentage'] = round(min(overlap_percentage, 100.0), 2)  # Ограничиваем 100%
            logger.debug(f"[{data['filename']}] Overlap percentage: {data['overlap_percentage']}%")
        else:
            data['overlap_percentage'] = 0.0

    logger.info("Расчет перекрытия завершен.")

def save_to_csv(data_list, headers):
    """Сохраняет данные в CSV файл."""
    logger.info(f"Запись данных в файл {OUTPUT_CSV}...")
    try:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for data in data_list:
                row = {key: data.get(key, '') for key in headers}
                # Форматируем дату и время
                row['shooting_time'] = data['shooting_time'].strftime('%Y-%m-%d %H:%M:%S') if data['shooting_time'] else ''
                writer.writerow(row)
        logger.info("Запись в CSV завершена успешно.")
    except Exception as e:
        logger.error(f"Ошибка при записи в CSV файл: {e}")

def save_to_excel(data_list, headers):
    """Сохраняет данные в Excel файл."""
    logger.info(f"Запись данных в файл {OUTPUT_EXCEL}...")
    try:
        df = pd.DataFrame(data_list)
        df = df[headers]
        df['shooting_time'] = df['shooting_time'].apply(
            lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if isinstance(x, datetime) else ''
        )
        df.to_excel(OUTPUT_EXCEL, index=False)
        logger.info("Запись в Excel завершена успешно.")
    except Exception as e:
        logger.error(f"Ошибка при записи в Excel файл: {e}")


def analyze_data(data_list):
    """Анализирует данные в соответствии с техническим заданием."""
    logger.info("Анализ данных в соответствии с техническим заданием...")
    total_images = len(data_list)
    if total_images == 0:
        logger.info("Нет данных для анализа.")
        return

    # Проверка перекрытия для высот выше 80 метров
    images_above_80m = [d for d in data_list if d['relative_altitude'] and d['relative_altitude'] > 80]
    images_below_80m = [d for d in data_list if d['relative_altitude'] and d['relative_altitude'] <= 80]

    # Анализ перекрытия
    overlap_criteria_above_80m = [d for d in images_above_80m if d['overlap_percentage'] is not None and d['overlap_percentage'] <= 15]
    overlap_criteria_below_80m = [d for d in images_below_80m if d['overlap_percentage'] is not None and d['overlap_percentage'] <= 50]

    percent_overlap_above_80m = len(overlap_criteria_above_80m) / len(images_above_80m) * 100 if images_above_80m else 0
    percent_overlap_below_80m = len(overlap_criteria_below_80m) / len(images_below_80m) * 100 if images_below_80m else 0

    logger.info(f"Процент фотографий с перекрытием не более 15% (высота > 80 м): {percent_overlap_above_80m:.2f}%")
    logger.info(f"Процент фотографий с перекрытием не более 50% (высота ≤ 80 м): {percent_overlap_below_80m:.2f}%")

    # Распределение фотографий по диапазонам высот
    total_in_ranges = 0
    for alt_range in ALTITUDE_RANGES:
        range_name = f"{alt_range[0]}-{alt_range[1]}"
        images_in_range = [d for d in data_list if d['relative_altitude'] and alt_range[0] <= d['relative_altitude'] <= alt_range[1]]
        percent_in_range = len(images_in_range) / total_images * 100
        total_in_ranges += len(images_in_range)
        logger.info(f"Процент фотографий на высоте от {alt_range[0]} до {alt_range[1]} м: {percent_in_range:.2f}%")

    # Проверка формата файлов и наличия высоты
    jpeg_images = [d for d in data_list if d['filename'].lower().endswith(('.jpg', '.jpeg'))]
    images_with_altitude = [d for d in data_list if d['relative_altitude'] is not None]

    percent_jpeg = len(jpeg_images) / total_images * 100
    percent_with_altitude = len(images_with_altitude) / total_images * 100

    logger.info(f"Процент фотографий в формате JPEG: {percent_jpeg:.2f}%")
    logger.info(f"Процент фотографий с информацией о высоте: {percent_with_altitude:.2f}%")

    logger.info("Обработка завершена.")


def visualize_overlaps(data_list, selected_sets=None):
    """
    Создает интерактивную визуализацию перекрытий фотографий с помощью Folium.
    """
    if VISUALIZE_ALL_IMAGES:
        logger.info("Создание визуализации для всех изображений...")
        create_visualization(data_list, 'overlaps_visualization.html', "Все изображения")

    if VISUALIZE_SELECTED_IMAGES and selected_sets:
        logger.info("Создание визуализации для выбранных изображений...")
        create_visualization_selected(selected_sets, 'selected_overlaps_visualization.html', "Выбранные изображения")


def create_visualization(data_to_visualize, output_html, title):
    # Проверяем наличие полигонов для визуализации
    if not any(data.get('polygon') for data in data_to_visualize):
        logger.info(f"Нет полигонов для визуализации {title.lower()}.")
        return

    # Определяем центр карты
    latitudes = [data['latitude'] for data in data_to_visualize if data['latitude'] is not None]
    longitudes = [data['longitude'] for data in data_to_visualize if data['longitude'] is not None]
    if not latitudes or not longitudes:
        logger.info(f"Нет корректных координат для визуализации {title.lower()}.")
        return
    center_lat = sum(latitudes) / len(latitudes)
    center_lon = sum(longitudes) / len(longitudes)

    # Создаем карту Folium
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    folium.TileLayer('cartodbpositron').add_to(m)
    folium.LayerControl().add_to(m)

    # Создаем трансформер для преобразования координат обратно в WGS84
    utm_epsg_code = data_to_visualize[0].get('utm_epsg_code')
    ref_x = data_to_visualize[0].get('ref_x', 0)
    ref_y = data_to_visualize[0].get('ref_y', 0)
    transformer_to_wgs84 = Transformer.from_crs(f"EPSG:{utm_epsg_code}", "EPSG:4326", always_xy=True)

    # Добавляем полигоны на карту
    for data in data_to_visualize:
        polygon = data.get('polygon')
        if not polygon:
            continue

        # Преобразуем координаты полигона обратно в WGS84
        exterior_coords = []
        for x, y in polygon.exterior.coords:
            lon, lat = transformer_to_wgs84.transform(x + ref_x, y + ref_y)
            exterior_coords.append((lat, lon))

        # Цвет полигона зависит от процента перекрытия
        overlap_percentage = data.get('overlap_percentage', 0)
        if overlap_percentage <= 15:
            color = 'green'
        elif overlap_percentage <= 50:
            color = 'orange'
        else:
            color = 'red'

        # Создаем полигон Folium
        folium_polygon = folium.Polygon(
            locations=exterior_coords,
            color=color,
            weight=1,
            fill=True,
            fill_opacity=0.5,
            popup=folium.Popup(
                f"Файл: {data['filename']}<br>"
                f"Перекрытие: {overlap_percentage}%<br>"
                f"Высота: {data.get('relative_altitude', 'N/A')} м",
                max_width=300
            )
        )
        folium_polygon.add_to(m)

    # Сохраняем карту в HTML файл
    m.save(output_html)
    logger.info(f"Интерактивная визуализация '{title}' сохранена в '{output_html}'.")


def create_visualization_selected(selected_sets, output_html, title):
    # Слияние всех сетов в один список для визуализации
    data_to_visualize = []
    for idx, image_set in enumerate(selected_sets):
        for data in image_set:
            data_copy = data.copy()
            data_copy['set_number'] = idx + 1  # Добавляем номер сета для отображения
            data_to_visualize.append(data_copy)

    # Проверяем наличие полигонов для визуализации
    if not any(data.get('polygon') for data in data_to_visualize):
        logger.info(f"Нет полигонов для визуализации {title.lower()}.")
        return

    # Определяем центр карты
    latitudes = [data['latitude'] for data in data_to_visualize if data['latitude'] is not None]
    longitudes = [data['longitude'] for data in data_to_visualize if data['longitude'] is not None]
    if not latitudes or not longitudes:
        logger.info(f"Нет корректных координат для визуализации {title.lower()}.")
        return
    center_lat = sum(latitudes) / len(latitudes)
    center_lon = sum(longitudes) / len(longitudes)

    # Создаем карту Folium
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    folium.TileLayer('cartodbpositron').add_to(m)
    folium.LayerControl().add_to(m)

    # Создаем трансформер для преобразования координат обратно в WGS84
    utm_epsg_code = data_to_visualize[0].get('utm_epsg_code')
    ref_x = data_to_visualize[0].get('ref_x', 0)
    ref_y = data_to_visualize[0].get('ref_y', 0)
    transformer_to_wgs84 = Transformer.from_crs(f"EPSG:{utm_epsg_code}", "EPSG:4326", always_xy=True)

    # Определяем цвета для каждого сета
    set_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # Добавляем полигоны на карту
    for data in data_to_visualize:
        polygon = data.get('polygon')
        if not polygon:
            continue

        # Преобразуем координаты полигона обратно в WGS84
        exterior_coords = []
        for x, y in polygon.exterior.coords:
            lon, lat = transformer_to_wgs84.transform(x + ref_x, y + ref_y)
            exterior_coords.append((lat, lon))

        set_number = data.get('set_number', 0)
        color = set_colors[(set_number - 1) % len(set_colors)]

        # Создаем полигон Folium
        folium_polygon = folium.Polygon(
            locations=exterior_coords,
            color=color,
            weight=1,
            fill=True,
            fill_opacity=0.5,
            popup=folium.Popup(
                f"Сет: {set_number}<br>"
                f"Файл: {data['filename']}<br>"
                f"Высота: {data.get('relative_altitude', 'N/A')} м",
                max_width=300
            )
        )
        folium_polygon.add_to(m)

    # Сохраняем карту в HTML файл
    m.save(output_html)
    logger.info(f"Интерактивная визуализация '{title}' сохранена в '{output_html}'.")


def copy_selected_images_sets(selected_sets: List[List[Dict[str, Any]]], base_destination_dir: str):
    """
    Копирует выбранные изображения в указанные папки для каждого сета и записывает список изображений в текстовый файл.

    :param selected_sets: Список сетов с выбранными изображениями.
    :param base_destination_dir: Базовый путь к папке назначения.
    """
    if not os.path.exists(base_destination_dir):
        try:
            os.makedirs(base_destination_dir)
            logger.info(f"Создана базовая папка назначения: {base_destination_dir}")
        except Exception as e:
            logger.error(f"Ошибка при создании папки {base_destination_dir}: {e}")
            return

    for idx, image_set in enumerate(selected_sets):
        set_dir = os.path.join(base_destination_dir, f"set_{idx+1}")
        if not os.path.exists(set_dir):
            try:
                os.makedirs(set_dir)
                logger.info(f"Создана папка для сета {idx+1}: {set_dir}")
            except Exception as e:
                logger.error(f"Ошибка при создании папки {set_dir}: {e}")
                continue

        image_filenames = []  # Список для хранения имен файлов изображений

        for img in image_set:
            filename = img['filename']
            source_path = img['file_path']
            destination_path = os.path.join(set_dir, filename)
            try:
                # Копируем файл без изменений
                with open(source_path, 'rb') as src_file:
                    with open(destination_path, 'wb') as dest_file:
                        dest_file.write(src_file.read())
                logger.info(f"Скопировано: {source_path} -> {destination_path}")
                image_filenames.append(filename)
            except Exception as e:
                logger.error(f"Ошибка при копировании файла {filename}: {e}")

        # Записываем список изображений в текстовый файл в папке сета
        try:
            list_file_path = os.path.join(set_dir, "image_list.txt")
            with open(list_file_path, 'w', encoding='utf-8') as f:
                for filename in image_filenames:
                    f.write(f"{filename}\n")
            logger.info(f"Список изображений для сета {idx+1} записан в файл: {list_file_path}")
        except Exception as e:
            logger.error(f"Ошибка при записи списка изображений для сета {idx+1}: {e}")


def select_images_sets(
    data_list: List[Dict[str, Any]],
    num_sets: int,
    num_images_per_set: int,
    overlap_percentage_between_sets: float,
    sort_by: str = 'relative_altitude',
    exclude_keyword: str = ''
) -> List[List[Dict[str, Any]]]:
    """
    Выбирает изображения и распределяет их по заданному количеству сетов (папок), обеспечивая
    заданный процент одинаковых фотографий (пересечений) между всеми парами сетов и заданное количество фотографий в каждом сете.
    """
    logger.info(
        f"Выбор изображений для {num_sets} сетов по {num_images_per_set} изображений в каждом сете с пересечением {overlap_percentage_between_sets}%."
    )

    # Исключаем изображения с заданным ключевым словом в названии файла или пути
    exclude_keyword_lower = exclude_keyword.lower()
    filtered_data = [
        d for d in data_list
        if exclude_keyword_lower not in d['filename'].lower()
           and exclude_keyword_lower not in d['relative_path'].lower()
           and d.get('polygon') is not None
    ]

    excluded_count = len(data_list) - len(filtered_data)
    if not filtered_data:
        logger.warning(f"Нет изображений после исключения ключевого слова '{exclude_keyword}'.")
        return []
    else:
        logger.info(f"Исключено {excluded_count} изображений по ключевому слову '{exclude_keyword}'.")

    # Расчет количества перекрывающихся и уникальных изображений
    overlap_count = int(num_images_per_set * overlap_percentage_between_sets / 100)
    num_pairs = num_sets * (num_sets - 1) // 2  # Количество пар сетов
    overlaps_per_set = overlap_count * (num_sets - 1)
    unique_images_per_set = num_images_per_set - overlaps_per_set

    if unique_images_per_set < 0:
        logger.warning("Процент перекрытия слишком высок для заданного количества сетов и изображений в сете.")
        return []

    total_overlapping_images = overlap_count * num_pairs
    total_unique_images_needed = unique_images_per_set * num_sets + total_overlapping_images

    if total_unique_images_needed > len(filtered_data):
        logger.warning("Недостаточно изображений для выполнения условий с заданными параметрами.")
        return []

    # Перемешиваем данные
    random.shuffle(filtered_data)

    used_images = set()
    unique_images_sets = [[] for _ in range(num_sets)]
    overlapping_images = {}

    image_pool = filtered_data.copy()

    # Формирование перекрывающихся изображений между всеми парами сетов
    logger.info("Формирование перекрывающихся изображений между всеми парами сетов...")
    from itertools import combinations

    for (i, j) in combinations(range(num_sets), 2):
        available_images = [img for img in image_pool if img['filename'] not in used_images]
        if len(available_images) < overlap_count:
            logger.warning("Недостаточно изображений для назначения перекрывающихся изображений между сетами.")
            return []
        selected_images = random.sample(available_images, overlap_count)
        overlapping_images[(i, j)] = selected_images
        used_images.update(img['filename'] for img in selected_images)
        image_pool = [img for img in image_pool if img['filename'] not in used_images]

    # Формирование уникальных изображений для каждого сета
    logger.info("Формирование уникальных изображений для каждого сета...")
    for i in range(num_sets):
        available_images = [img for img in image_pool if img['filename'] not in used_images]
        if len(available_images) < unique_images_per_set:
            logger.warning("Недостаточно изображений для назначения уникальных изображений для каждого сета.")
            return []
        selected_images = random.sample(available_images, unique_images_per_set)
        unique_images_sets[i] = selected_images
        used_images.update(img['filename'] for img in selected_images)
        image_pool = [img for img in image_pool if img['filename'] not in used_images]

    # Сборка каждого сета
    selected_sets = []
    for i in range(num_sets):
        current_set = unique_images_sets[i].copy()
        # Добавляем перекрывающиеся изображения с другими сетами
        for j in range(num_sets):
            if i == j:
                continue
            if i < j:
                overlap_imgs = overlapping_images.get((i, j), [])
            else:
                overlap_imgs = overlapping_images.get((j, i), [])
            current_set.extend(overlap_imgs)
        # Проверяем и удаляем возможные дубликаты (на всякий случай)
        current_set = list({img['filename']: img for img in current_set}.values())
        if len(current_set) != num_images_per_set:
            logger.warning(f"Сет {i+1} не содержит корректное количество изображений.")
            return []
        # Сортируем текущий сет
        current_set.sort(key=lambda x: x.get(sort_by, 0) or 0)
        selected_sets.append(current_set)

    logger.info("Формирование сетов завершено.")
    return selected_sets


def create_overlap_map(selected_sets: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Создает карту пересечений для отобранных фотографий с подсчетом пересечений
    внутри всех сетов и внутри каждого сета.

    :param selected_sets: Список сетов с выбранными изображениями.
    :return: Словарь с информацией о пересечениях.
    """
    logger.info("Создание карты пересечений между сетами...")

    overlap_map = {}
    total_images = set()

    # Собираем все уникальные имена файлов
    for idx, image_set in enumerate(selected_sets):
        set_number = idx + 1
        image_filenames = set(img['filename'] for img in image_set)
        total_images.update(image_filenames)
        overlap_map[f"set_{set_number}"] = {
            "images": list(image_filenames),  # Преобразуем set в list
            "overlaps_within_set": {},
            "overlaps_with_other_sets": {}
        }

    # Вычисляем пересечения внутри каждого сета (если нужно)
    for key, value in overlap_map.items():
        images_in_set = value['images']
        # В данном случае, так как в каждом сете изображения уникальны после отбора, пересечений внутри сета нет
        value['overlaps_within_set'] = 0  # Для консистентности

    # Вычисляем пересечения между сетами
    for idx1 in range(len(selected_sets)):
        set_num1 = idx1 + 1
        images_set1 = set(overlap_map[f"set_{set_num1}"]['images'])
        for idx2 in range(idx1 + 1, len(selected_sets)):
            set_num2 = idx2 + 1
            images_set2 = set(overlap_map[f"set_{set_num2}"]['images'])
            overlap_images = images_set1.intersection(images_set2)
            overlap_count = len(overlap_images)
            overlap_percentage = (overlap_count / NUM_IMAGES_PER_SET) * 100
            logger.info(f"Пересечение между сетом {set_num1} и сетом {set_num2}: {overlap_count} изображений ({overlap_percentage:.2f}%)")
            overlap_map[f"set_{set_num1}"]['overlaps_with_other_sets'][f"set_{set_num2}"] = overlap_count
            overlap_map[f"set_{set_num2}"]['overlaps_with_other_sets'][f"set_{set_num1}"] = overlap_count

    logger.info("Карта пересечений создана.")

    return overlap_map


def main():
    data_list = []
    hash_set = set()

    # Собираем список всех файлов для обработки
    logger.info("Сканирование каталогов для поиска изображений...")
    files_to_process = []
    for folder_name, _, filenames in os.walk(ROOT_DIR):
        for filename in filenames:
            if filename.lower().endswith(('.jpg', '.jpeg')):
                file_path = os.path.join(folder_name, filename)
                relative_path = os.path.relpath(file_path, ROOT_DIR)
                files_to_process.append((file_path, relative_path, filename))
    total_files = len(files_to_process)
    logger.info(f"Найдено {total_files} изображений для обработки.")

    if total_files == 0:
        logger.info("Нет изображений для обработки.")
        return

    # Ограничиваем количество файлов для обработки
    if MAX_FILES_TO_PROCESS and total_files > MAX_FILES_TO_PROCESS:
        files_to_process = files_to_process[:MAX_FILES_TO_PROCESS]
        logger.info(f"Ограничение на количество обрабатываемых изображений: {MAX_FILES_TO_PROCESS}")

    # Собираем все координаты для определения области покрытия
    logger.info("Извлечение координат из фотографий...")
    latitudes = []
    longitudes = []
    for file_info in files_to_process:
        file_path, relative_path, filename = file_info
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            latitude, longitude, altitude = parse_gps_and_altitude(tags)
            if latitude is not None and longitude is not None:
                latitudes.append(latitude)
                longitudes.append(longitude)

    if not latitudes or not longitudes:
        logger.info("Не удалось получить координаты из фотографий. Проверьте наличие GPS данных.")
        return

    # Определяем bounding box
    min_lat, max_lat = min(latitudes), max(latitudes)
    min_lon, max_lon = min(longitudes), max(longitudes)
    logger.info(f"Определена область покрытия: ({min_lat}, {min_lon}) - ({max_lat}, {max_lon})")

    # Вычисляем средние широту и долготу
    mean_lat = sum(latitudes) / len(latitudes)
    mean_lon = sum(longitudes) / len(longitudes)

    # Вычисляем номер зоны UTM
    utm_zone_number = int((mean_lon + 180) / 6) + 1
    hemisphere = 'north' if mean_lat >= 0 else 'south'
    logger.info(f"Вычислена зона UTM: {utm_zone_number}, полушарие: {hemisphere}")

    # Используем стандартные EPSG-коды для UTM
    if hemisphere == 'north':
        utm_epsg_code = 32600 + utm_zone_number  # EPSG-коды для северного полушария
    else:
        utm_epsg_code = 32700 + utm_zone_number  # EPSG-коды для южного полушария

    # Инициализируем преобразователь координат с использованием CRS
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg_code}", always_xy=True)

    # Устанавливаем референсную точку
    global_utm_epsg_code = utm_epsg_code
    ref_lat = latitudes[0]
    ref_lon = longitudes[0]
    if ref_lat is None or ref_lon is None:
        logger.info("Референсная точка не имеет корректных координат.")
        return
    ref_x, ref_y = transformer.transform(ref_lon, ref_lat)

    # Параллельная обработка файлов
    logger.info("Начинаем параллельную обработку изображений...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for file_info in files_to_process:
            futures.append(
                executor.submit(process_file, file_info, transformer, ref_x, ref_y, hash_set, global_utm_epsg_code))

        for future in as_completed(futures):
            result = future.result()
            if result:
                data_list.append(result)

    if not data_list:
        logger.info("Нет данных для обработки после параллельной обработки.")
        return

    # Сортируем данные по времени съемки
    logger.info("Сортировка данных по времени съемки...")
    data_list.sort(key=lambda x: x['shooting_time'] or datetime.min)

    # Получаем высоты местности и вычисляем relative_altitude
    fetch_elevations(data_list)

    # Вычисляем дополнительные параметры
    logger.info("Вычисление дополнительных параметров...")
    calculate_additional_parameters(data_list)

    # Вычисляем перекрытие
    calculate_overlap(data_list)

    # Выбор изображений и создание сетов
    selected_sets = select_images_sets(
        data_list,
        num_sets=NUM_SETS_TO_CREATE,
        num_images_per_set=NUM_IMAGES_PER_SET,
        overlap_percentage_between_sets=OVERLAP_PERCENTAGE_BETWEEN_SETS,
        sort_by='relative_altitude',
        exclude_keyword=EXCLUDE_KEYWORD
    )

    if selected_sets:
        # Копирование выбранных изображений
        copy_selected_images_sets(selected_sets, DESTINATION_DIR)
        logger.info(f"Избранные изображения скопированы в папку: {DESTINATION_DIR}")

        # Создание карты пересечений
        overlap_map = create_overlap_map(selected_sets)
        # Сериализуем overlap_map в JSON
        overlap_map_json = json.dumps(overlap_map, indent=2, ensure_ascii=False)
        logger.info("Карта пересечений:")
        logger.info(overlap_map_json)

    else:
        logger.info("Нет изображений для копирования.")

    # Визуализируем перекрытия
    visualize_overlaps(data_list, selected_sets)

    # Записываем данные
    headers = [
        'filename', 'relative_path', 'latitude', 'longitude', 'altitude',
        'terrain_elevation', 'relative_altitude', 'fov_horizontal', 'fov_vertical',
        'shooting_time', 'image_width', 'image_height', 'width_meters',
        'height_meters', 'resolution_cm_per_pixel', 'distance_to_previous',
        'size_in_pixels', 'overlap_percentage', 'digital_zoom_ratio'
    ]
    save_to_csv(data_list, headers)
    save_to_excel(data_list, headers)

    # Анализ данных
    analyze_data(data_list)


if __name__ == '__main__':
    main()