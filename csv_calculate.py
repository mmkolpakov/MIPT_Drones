import os
import exifread
import math
import csv
import hashlib
import pandas as pd
from datetime import datetime
from shapely.geometry import box
from shapely.strtree import STRtree
from pyproj import Proj, Transformer
from dateutil import parser
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Настраиваемые параметры
ROOT_DIR = r'C:\Users\mkolp\OneDrive\Документы\Датасет'  # Корневой каталог с изображениями
OUTPUT_CSV = 'output.csv'  # Имя выходного CSV файла
OUTPUT_EXCEL = 'output.xlsx'  # Имя выходного Excel файла
SENSOR_WIDTH_MM = 36.0  # Ширина сенсора в мм (для камеры Zenmuse P1)
SENSOR_HEIGHT_MM = 24.0  # Высота сенсора в мм (для камеры Zenmuse P1)
OBJECT_SIZE_M = 1.0  # Размер объекта в метрах для расчета size_in_pixels
MAX_WORKERS = 4  # Максимальное количество потоков для параллельной обработки

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
    return 2 * math.degrees(math.atan(sensor_size_mm / (2 * focal_length_mm)))

def compute_file_hash(file_path):
    """Вычисляет хеш файла для обнаружения дубликатов."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as afile:
        for chunk in iter(lambda: afile.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def process_file(file_info, transformer, ref_x, ref_y, hash_set):
    """Обрабатывает один файл изображения."""
    file_path, relative_path, filename = file_info
    data = {}
    try:
        # Вычисляем хеш файла для обнаружения дубликатов
        file_hash = compute_file_hash(file_path)
        if file_hash in hash_set:
            print(f"Предупреждение: Обнаружен дубликат файла {filename}. Запись будет пропущена.")
            return None  # Пропускаем обработку этого файла
        else:
            hash_set.add(file_hash)

        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)

            if not tags:
                print(f"Предупреждение: EXIF-данные не найдены в файле {filename}.")
                return None

            # Извлекаем данные GPS и высоту
            latitude, longitude, altitude = parse_gps_and_altitude(tags)

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
            }

            # Преобразование координат в метры относительно референсной точки
            if None not in (latitude, longitude):
                x, y = transformer.transform(longitude, latitude)
                data['x'] = x - ref_x
                data['y'] = y - ref_y
            else:
                data['x'] = data['y'] = None

            return data
    except Exception as e:
        print(f"Ошибка при обработке файла {filename}: {e}")
        return None

def fetch_elevations(data_list):
    """Получает высоты местности для всех точек и обновляет data_list."""
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
        print("Нет координат для получения высот местности.")
        return

    # Отправляем POST запрос к API
    print("Получение высот местности для всех точек...")
    try:
        url = 'https://api.open-elevation.com/api/v1/lookup'
        headers = {'Content-Type': 'application/json'}
        data = {'locations': coords}
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            results = response.json()['results']
            if len(results) != len(data_with_coords):
                print("Количество полученных высот не соответствует количеству точек.")
                return
            # Обновляем data_list с высотами
            for idx, result in enumerate(results):
                elevation = result.get('elevation')
                data_with_coords[idx]['terrain_elevation'] = elevation
                altitude = data_with_coords[idx].get('altitude')
                if elevation is not None and altitude is not None:
                    data_with_coords[idx]['relative_altitude'] = altitude - elevation
                else:
                    data_with_coords[idx]['relative_altitude'] = None
        else:
            print(f"Ошибка запроса к API open-elevation: {response.status_code}")
    except Exception as e:
        print(f"Ошибка при получении высот местности: {e}")

def calculate_additional_parameters(data_list):
    """Вычисляет дополнительные параметры для каждого изображения."""
    previous_data = None
    for data in data_list:
        effective_focal_length = data['effective_focal_length']
        relative_altitude = data['relative_altitude']
        image_width = data['image_width']
        image_height = data['image_height']
        latitude = data['latitude']
        longitude = data['longitude']

        if effective_focal_length is None or effective_focal_length == 0:
            print(f"Предупреждение: Недостаточно данных для вычисления параметров изображения {data['filename']}.")
            continue

        # Углы обзора
        data['fov_horizontal'] = calculate_field_of_view(SENSOR_WIDTH_MM, effective_focal_length)
        data['fov_vertical'] = calculate_field_of_view(SENSOR_HEIGHT_MM, effective_focal_length)

        # Размеры на земле и разрешение
        if None not in (relative_altitude, image_width, image_height):
            # Приводим всё к метрам
            sensor_width_m = SENSOR_WIDTH_MM / 1000  # мм в метры
            sensor_height_m = SENSOR_HEIGHT_MM / 1000
            focal_length_m = effective_focal_length / 1000
            altitude_m = relative_altitude  # Используем относительную высоту

            gsd_horizontal = (altitude_m * sensor_width_m) / (focal_length_m * image_width)
            gsd_vertical = (altitude_m * sensor_height_m) / (focal_length_m * image_height)
            width_meters = gsd_horizontal * image_width
            height_meters = gsd_vertical * image_height
            resolution_cm_per_pixel = ((gsd_horizontal + gsd_vertical) / 2) * 100  # Метры в сантиметры
            size_in_pixels = (OBJECT_SIZE_M * 100) / resolution_cm_per_pixel  # Для объекта размером 1 метр

            data['width_meters'] = round(width_meters, 3)
            data['height_meters'] = round(height_meters, 3)
            data['resolution_cm_per_pixel'] = round(resolution_cm_per_pixel, 3)
            data['size_in_pixels'] = round(size_in_pixels, 2)
        else:
            data['width_meters'] = data['height_meters'] = data['resolution_cm_per_pixel'] = data['size_in_pixels'] = None

        # Расстояние до предыдущего снимка
        if previous_data and None not in (latitude, longitude, previous_data['latitude'], previous_data['longitude']):
            x1, y1 = data['x'], data['y']
            x2, y2 = previous_data['x'], previous_data['y']
            distance = math.hypot(x2 - x1, y2 - y1)
            data['distance_to_previous'] = round(distance, 2)
        else:
            data['distance_to_previous'] = None

        previous_data = data

def calculate_overlap(data_list):
    """Вычисляет пересечения между изображениями."""
    print("Создание полигонов для каждого изображения...")
    data_with_polygons = []
    polygons = []
    for data in data_list:
        if None not in (data['x'], data['y'], data['width_meters'], data['height_meters']):
            half_width = data['width_meters'] / 2
            half_height = data['height_meters'] / 2
            polygon = box(
                data['x'] - half_width,
                data['y'] - half_height,
                data['x'] + half_width,
                data['y'] + half_height
            )
            data['polygon'] = polygon
            data['area'] = data['width_meters'] * data['height_meters']
            data['overlap_area'] = 0.0
            data_with_polygons.append(data)
            polygons.append(polygon)
        else:
            data['polygon'] = None
            data['area'] = None
            data['overlap_area'] = 0.0

    if data_with_polygons:
        tree = STRtree(polygons)
        for idx_data, data in enumerate(data_with_polygons):
            query_geom = data['polygon']
            possible_indices = tree.query(query_geom)
            for idx in possible_indices:
                if idx == idx_data:
                    continue
                other_geom = polygons[idx]
                other_data = data_with_polygons[idx]
                if query_geom.intersects(other_geom):
                    intersection = query_geom.intersection(other_geom)
                    overlap_area = intersection.area
                    data['overlap_area'] += overlap_area
                    other_data['overlap_area'] += overlap_area
    else:
        print("No polygons available for overlap calculation.")
    for data in data_list:
        if data['overlap_area'] is not None and data['area'] is not None and data['area'] != 0:
            data['overlap_percentage'] = round((data['overlap_area'] / data['area']) * 100, 2)
        else:
            data['overlap_percentage'] = None

def save_to_csv(data_list, headers):
    """Сохраняет данные в CSV файл."""
    print(f"Запись данных в файл {OUTPUT_CSV}...")
    try:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for data in data_list:
                row = {key: data.get(key, '') for key in headers}
                # Форматируем дату и время
                row['shooting_time'] = data['shooting_time'].strftime('%Y-%m-%d %H:%M:%S') if data['shooting_time'] else ''
                writer.writerow(row)
        print("Запись в CSV завершена успешно.")
    except Exception as e:
        print(f"Ошибка при записи в CSV файл: {e}")

def save_to_excel(data_list, headers):
    """Сохраняет данные в Excel файл."""
    print(f"Запись данных в файл {OUTPUT_EXCEL}...")
    try:
        df = pd.DataFrame(data_list)
        df = df[headers]
        df['shooting_time'] = df['shooting_time'].apply(
            lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if isinstance(x, datetime) else ''
        )
        df.to_excel(OUTPUT_EXCEL, index=False)
        print("Запись в Excel завершена успешно.")
    except Exception as e:
        print(f"Ошибка при записи в Excel файл: {e}")

def analyze_data(data_list):
    """Анализирует данные в соответствии с техническим заданием."""
    print("Анализ данных в соответствии с техническим заданием...")
    total_images = len(data_list)
    if total_images == 0:
        print("Нет данных для анализа.")
        return

    # Проверка перекрытия для высот выше 80 метров
    images_above_80m = [d for d in data_list if d['relative_altitude'] and d['relative_altitude'] > 80]
    images_below_80m = [d for d in data_list if d['relative_altitude'] and d['relative_altitude'] <= 80]

    # Анализ перекрытия
    overlap_criteria_above_80m = [d for d in images_above_80m if d['overlap_percentage'] is not None and d['overlap_percentage'] <= 15]
    overlap_criteria_below_80m = [d for d in images_below_80m if d['overlap_percentage'] is not None and d['overlap_percentage'] <= 50]

    percent_overlap_above_80m = len(overlap_criteria_above_80m) / len(images_above_80m) * 100 if images_above_80m else 0
    percent_overlap_below_80m = len(overlap_criteria_below_80m) / len(images_below_80m) * 100 if images_below_80m else 0

    print(f"Процент фотографий с перекрытием не более 15% (высота > 80 м): {percent_overlap_above_80m:.2f}%")
    print(f"Процент фотографий с перекрытием не более 50% (высота ≤ 80 м): {percent_overlap_below_80m:.2f}%")

    # Распределение фотографий по диапазонам высот
    range_20_80 = [d for d in data_list if d['relative_altitude'] and 20 <= d['relative_altitude'] <= 80]
    range_80_140 = [d for d in data_list if d['relative_altitude'] and 80 < d['relative_altitude'] <= 140]
    range_140_200 = [d for d in data_list if d['relative_altitude'] and 140 < d['relative_altitude'] <= 200]

    percent_20_80 = len(range_20_80) / total_images * 100
    percent_80_140 = len(range_80_140) / total_images * 100
    percent_140_200 = len(range_140_200) / total_images * 100

    print(f"Процент фотографий на высоте от 20 до 80 м: {percent_20_80:.2f}%")
    print(f"Процент фотографий на высоте от 80 до 140 м: {percent_80_140:.2f}%")
    print(f"Процент фотографий на высоте от 140 до 200 м: {percent_140_200:.2f}%")

    # Проверка формата файлов и наличия высоты
    jpeg_images = [d for d in data_list if d['filename'].lower().endswith(('.jpg', '.jpeg'))]
    images_with_altitude = [d for d in data_list if d['relative_altitude'] is not None]

    percent_jpeg = len(jpeg_images) / total_images * 100
    percent_with_altitude = len(images_with_altitude) / total_images * 100

    print(f"Процент фотографий в формате JPEG: {percent_jpeg:.2f}%")
    print(f"Процент фотографий с информацией о высоте: {percent_with_altitude:.2f}%")

    print("Обработка завершена.")

def main():
    data_list = []
    hash_set = set()
    duplicates_found = False

    # Собираем список всех файлов для обработки
    print("Сканирование каталогов для поиска изображений...")
    files_to_process = []
    for folder_name, _, filenames in os.walk(ROOT_DIR):
        for filename in filenames:
            if filename.lower().endswith(('.jpg', '.jpeg')):
                file_path = os.path.join(folder_name, filename)
                relative_path = os.path.relpath(file_path, ROOT_DIR)
                files_to_process.append((file_path, relative_path, filename))
    total_files = len(files_to_process)
    print(f"Найдено {total_files} изображений для обработки.")

    if total_files == 0:
        print("Нет изображений для обработки.")
        return

    # Собираем все координаты для определения области покрытия
    print("Извлечение координат из фотографий...")
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
        print("Не удалось получить координаты из фотографий. Проверьте наличие GPS данных.")
        return

    # Определяем bounding box
    min_lat, max_lat = min(latitudes), max(latitudes)
    min_lon, max_lon = min(longitudes), max(longitudes)
    print(f"Определена область покрытия: ({min_lat}, {min_lon}) - ({max_lat}, {max_lon})")

    # Вычисляем средние широту и долготу
    mean_lat = sum(latitudes) / len(latitudes)
    mean_lon = sum(longitudes) / len(longitudes)

    # Вычисляем номер зоны UTM
    utm_zone_number = int((mean_lon + 180) / 6) + 1
    hemisphere = 'north' if mean_lat >= 0 else 'south'
    print(f"Вычислена зона UTM: {utm_zone_number}, полушарие: {hemisphere}")

    # Инициализируем преобразователь координат без использования CRS
    proj_wgs84 = Proj('epsg:4326')
    # Задаем проекцию UTM с учетом полушария
    if hemisphere == 'north':
        proj_utm = Proj(proj='utm', zone=utm_zone_number, ellps='WGS84', datum='WGS84', units='m')
    else:
        proj_utm = Proj(proj='utm', zone=utm_zone_number, ellps='WGS84', datum='WGS84', units='m', south=True)
    transformer = Transformer.from_proj(proj_wgs84, proj_utm)

    # Устанавливаем референсную точку
    ref_lat = latitudes[0]
    ref_lon = longitudes[0]
    if ref_lat is None or ref_lon is None:
        print("Референсная точка не имеет корректных координат.")
        return
    ref_x, ref_y = transformer.transform(ref_lon, ref_lat)

    # Параллельная обработка файлов
    print("Начинаем параллельную обработку изображений...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for file_info in files_to_process:
            futures.append(executor.submit(process_file, file_info, transformer, ref_x, ref_y, hash_set))

        for future in as_completed(futures):
            result = future.result()
            if result:
                data_list.append(result)

    if not data_list:
        print("Нет данных для обработки после параллельной обработки.")
        return

    # Сортируем данные по времени съемки
    print("Сортировка данных по времени съемки...")
    data_list.sort(key=lambda x: x['shooting_time'] or datetime.min)

    # Получаем высоты местности и вычисляем relative_altitude
    fetch_elevations(data_list)

    # Вычисляем дополнительные параметры
    print("Вычисление дополнительных параметров...")
    calculate_additional_parameters(data_list)

    # Вычисляем перекрытие
    calculate_overlap(data_list)

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
