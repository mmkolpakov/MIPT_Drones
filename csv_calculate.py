import os
import exifread
import math
import csv
import hashlib
import pandas as pd
from datetime import datetime

# Настраиваемые параметры
ROOT_DIR = r'C:\Users\mkolp\OneDrive\Документы\Датасет'  # Корневой каталог с изображениями
OUTPUT_CSV = 'output.csv'  # Имя выходного CSV файла
OUTPUT_EXCEL = 'output.xlsx'  # Имя выходного Excel файла
SENSOR_WIDTH_MM = 36.0  # Ширина сенсора в мм (для камеры Zenmuse P1)
SENSOR_HEIGHT_MM = 24.0  # Высота сенсора в мм (для камеры Zenmuse P1)

# Константы для математических вычислений
EARTH_RADIUS_M = 6371000  # Радиус Земли в метрах
METER_TO_CM = 100  # Коэффициент перевода метров в сантиметры
CM_TO_METER = 0.01  # Коэффициент перевода сантиметров в метры
OBJECT_SIZE_M = 1.0  # Размер объекта в метрах для расчета size_in_pixels

def get_decimal_from_dms(dms, ref):
    """Преобразует координаты GPS из формата DMS в десятичные градусы."""
    degrees = dms.values[0].num / dms.values[0].den
    minutes = dms.values[1].num / dms.values[1].den
    seconds = dms.values[2].num / dms.values[2].den

    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def parse_gps(tags):
    """Извлекает информацию GPS из EXIF тегов."""
    latitude = longitude = altitude = None
    gps_tags = {
        'GPS GPSLatitude': 'latitude',
        'GPS GPSLatitudeRef': 'lat_ref',
        'GPS GPSLongitude': 'longitude',
        'GPS GPSLongitudeRef': 'lon_ref',
        'GPS GPSAltitude': 'altitude',
        'GPS GPSAltitudeRef': 'altitude_ref',
    }

    gps_data = {}
    for tag, key in gps_tags.items():
        if tag in tags:
            gps_data[key] = tags[tag]

    if 'latitude' in gps_data and 'lat_ref' in gps_data and \
       'longitude' in gps_data and 'lon_ref' in gps_data and \
       'altitude' in gps_data:
        lat = gps_data['latitude']
        lat_ref = gps_data['lat_ref'].values
        lon = gps_data['longitude']
        lon_ref = gps_data['lon_ref'].values
        alt = gps_data['altitude']
        alt_ref = gps_data.get('altitude_ref', None)

        latitude = get_decimal_from_dms(lat, lat_ref)
        longitude = get_decimal_from_dms(lon, lon_ref)
        altitude = alt.values[0].num / alt.values[0].den

        # Если AltitudeRef существует и равно 1, высота ниже уровня моря
        if alt_ref and alt_ref.values[0] == 1:
            altitude = -altitude

    return latitude, longitude, altitude

def haversine(lat1, lon1, lat2, lon2):
    """Вычисляет расстояние между двумя GPS координатами по формуле гаверсинусов."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    meters = EARTH_RADIUS_M * c
    return meters

def calculate_field_of_view(sensor_size_mm, focal_length_mm):
    """Вычисляет угол обзора в градусах."""
    return 2 * math.degrees(math.atan(sensor_size_mm / (2 * focal_length_mm)))

def compute_file_hash(file_path):
    """Вычисляет хеш файла для обнаружения дубликатов."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as afile:
        buf = afile.read()
        hasher.update(buf)
    return hasher.hexdigest()

def main():
    data_list = []
    hash_set = set()  # Множество для хранения хешей файлов
    duplicates_found = False  # Флаг наличия дубликатов

    # Собираем список всех файлов для обработки
    print("Сканирование каталогов для поиска изображений...")
    files_to_process = []
    for folder_name, _, filenames in os.walk(ROOT_DIR):
        folder = os.path.basename(folder_name)
        for filename in filenames:
            if filename.lower().endswith(('.jpg', '.jpeg')):
                file_path = os.path.join(folder_name, filename)
                files_to_process.append((file_path, folder, filename))
    total_files = len(files_to_process)
    print(f"Найдено {total_files} изображений для обработки.")

    # Обрабатываем каждый файл
    print("Начинаем обработку изображений...")
    for idx, (file_path, folder, filename) in enumerate(files_to_process, 1):
        print(f"Обработка файла {idx}/{total_files}: {filename}")
        try:
            # Вычисляем хеш файла для обнаружения дубликатов
            file_hash = compute_file_hash(file_path)
            if file_hash in hash_set:
                print(f"Предупреждение: Обнаружен дубликат файла {filename}. Запись будет пропущена.")
                duplicates_found = True
                continue  # Пропускаем обработку этого файла
            else:
                hash_set.add(file_hash)

            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)

                # Извлекаем данные GPS
                latitude, longitude, altitude = parse_gps(tags)

                # Извлекаем время съемки
                datetime_original = tags.get('EXIF DateTimeOriginal', tags.get('Image DateTime'))
                if datetime_original:
                    datetime_original = datetime_original.values
                else:
                    datetime_original = None

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

                data = {
                    'filename': filename,
                    'folder': folder,
                    'latitude': latitude,
                    'longitude': longitude,
                    'altitude': altitude,
                    'shooting_time': datetime_original,
                    'image_width': image_width,
                    'image_height': image_height,
                    'focal_length': focal_length,
                }
                data_list.append(data)
        except Exception as e:
            print(f"Ошибка при обработке файла {filename}: {e}")

    if duplicates_found:
        print("Обнаружены дубликаты файлов. Проверьте предупреждения выше.")

    # Преобразуем время съемки в объект datetime и сортируем данные
    print("Сортировка данных по времени съемки...")
    for data in data_list:
        dt_str = data['shooting_time']
        if dt_str:
            try:
                data['shooting_time'] = datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
            except ValueError:
                # Обработка случая, если формат времени отличается
                try:
                    data['shooting_time'] = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    data['shooting_time'] = None
        else:
            data['shooting_time'] = None

    data_list.sort(key=lambda x: x['shooting_time'] or datetime.min)

    # Вычисляем дополнительные параметры
    print("Вычисление дополнительных параметров...")
    previous_data = None

    for idx, data in enumerate(data_list, 1):
        focal_length = data['focal_length']
        altitude = data['altitude']
        image_width = data['image_width']
        image_height = data['image_height']

        # Угол обзора
        if focal_length:
            data['fov_horizontal'] = calculate_field_of_view(SENSOR_WIDTH_MM, focal_length)
            data['fov_vertical'] = calculate_field_of_view(SENSOR_HEIGHT_MM, focal_length)
        else:
            data['fov_horizontal'] = data['fov_vertical'] = None

        # Размеры на земле и разрешение
        if None not in (focal_length, altitude, image_width, image_height):
            altitude_cm = altitude * METER_TO_CM  # Переводим в см
            gsd_horizontal = (altitude_cm * SENSOR_WIDTH_MM) / (focal_length * image_width)
            gsd_vertical = (altitude_cm * SENSOR_HEIGHT_MM) / (focal_length * image_height)
            width_meters = (gsd_horizontal * image_width) * CM_TO_METER  # Переводим см в метры
            height_meters = (gsd_vertical * image_height) * CM_TO_METER
            resolution_cm_per_pixel = (gsd_horizontal + gsd_vertical) / 2
            size_in_pixels = (OBJECT_SIZE_M * METER_TO_CM) / resolution_cm_per_pixel  # Для объекта размером 1 метр

            data['width_meters'] = round(width_meters, 3)
            data['height_meters'] = round(height_meters, 3)
            data['resolution_cm_per_pixel'] = round(resolution_cm_per_pixel, 3)
            data['size_in_pixels'] = round(size_in_pixels, 2)
        else:
            data['width_meters'] = data['height_meters'] = data['resolution_cm_per_pixel'] = data['size_in_pixels'] = None

        # Расстояние до предыдущего снимка
        if previous_data and None not in (data['latitude'], data['longitude'], previous_data['latitude'], previous_data['longitude']):
            distance = haversine(previous_data['latitude'], previous_data['longitude'], data['latitude'], data['longitude'])
            data['distance_to_previous'] = round(distance, 2)
        else:
            data['distance_to_previous'] = None

        previous_data = data

    # Записываем данные в CSV
    print(f"Запись данных в файл {OUTPUT_CSV}...")
    headers = [
        'filename', 'folder', 'latitude', 'longitude', 'altitude', 'fov_horizontal',
        'shooting_time', 'image_width', 'image_height', 'width_meters', 'height_meters',
        'resolution_cm_per_pixel', 'distance_to_previous', 'size_in_pixels'
    ]

    try:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for data in data_list:
                row = {
                    'filename': data['filename'],
                    'folder': data['folder'],
                    'latitude': data['latitude'],
                    'longitude': data['longitude'],
                    'altitude': data['altitude'],
                    'fov_horizontal': data['fov_horizontal'],
                    'shooting_time': data['shooting_time'].strftime('%Y-%m-%d %H:%M:%S') if data['shooting_time'] else '',
                    'image_width': data['image_width'],
                    'image_height': data['image_height'],
                    'width_meters': data['width_meters'],
                    'height_meters': data['height_meters'],
                    'resolution_cm_per_pixel': data['resolution_cm_per_pixel'],
                    'distance_to_previous': data['distance_to_previous'],
                    'size_in_pixels': data['size_in_pixels'],
                }
                writer.writerow(row)
        print("Запись в CSV завершена успешно.")
    except Exception as e:
        print(f"Ошибка при записи в CSV файл: {e}")

    # Записываем данные в Excel
    print(f"Запись данных в файл {OUTPUT_EXCEL}...")
    try:
        # Преобразуем список словарей в DataFrame
        df = pd.DataFrame(data_list)
        # Переупорядочиваем столбцы согласно заголовкам
        df = df[headers]
        # Форматируем время съемки
        df['shooting_time'] = df['shooting_time'].apply(
            lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if isinstance(x, datetime) else ''
        )
        # Записываем в Excel
        df.to_excel(OUTPUT_EXCEL, index=False)
        print("Запись в Excel завершена успешно.")
    except Exception as e:
        print(f"Ошибка при записи в Excel файл: {e}")

    print("Обработка завершена.")

if __name__ == '__main__':
    main()
