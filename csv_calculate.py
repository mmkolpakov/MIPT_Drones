import os
import exifread
import math
import csv
import pandas as pd
from datetime import datetime

# Настраиваемые параметры
ROOT_DIR = r'C:\Users\mkolp\OneDrive\Документы\Датасет'  # Корневой каталог с изображениями
OUTPUT_CSV = 'output.csv'  # Имя выходного CSV файла
OUTPUT_EXCEL = 'output.xlsx'  # Имя выходного Excel файла
ALTITUDE_NUMBERS = ['140', '160', '180', '200']  # Номера папок по высоте съемки
SENSOR_WIDTH_MM = 36.0  # Ширина сенсора в мм
SENSOR_HEIGHT_MM = 24.0  # Высота сенсора в мм

def get_decimal_from_dms(dms, ref):
    """Преобразует координаты GPS из формата DMS в градусы."""
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
    gps_tags = ['GPS GPSLatitude', 'GPS GPSLatitudeRef',
                'GPS GPSLongitude', 'GPS GPSLongitudeRef', 'GPS GPSAltitude']

    if all(tag in tags for tag in gps_tags):
        lat = tags['GPS GPSLatitude']
        lat_ref = tags['GPS GPSLatitudeRef'].values
        lon = tags['GPS GPSLongitude']
        lon_ref = tags['GPS GPSLongitudeRef'].values
        alt = tags['GPS GPSAltitude']

        latitude = get_decimal_from_dms(lat, lat_ref)
        longitude = get_decimal_from_dms(lon, lon_ref)
        altitude = alt.values[0].num / alt.values[0].den
    return latitude, longitude, altitude

def haversine(lat1, lon1, lat2, lon2):
    """Вычисляет расстояние между двумя GPS координатами по формуле."""
    R = 6371000  # Радиус Земли в метрах
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    meters = R * c
    return meters

def calculate_field_of_view(sensor_size_mm, focal_length_mm):
    """Вычисляет угол обзора в градусах."""
    return 2 * math.degrees(math.atan(sensor_size_mm / (2 * focal_length_mm)))

def main():
    data_list = []

    # Собираем список всех файлов для обработки
    print("Сканирование каталогов для поиска изображений...")
    files_to_process = []
    for folder_name, _, filenames in os.walk(ROOT_DIR):
        # Получаем список всех частей пути к папке
        folder_parts = os.path.normpath(folder_name).split(os.sep)
        # Проверяем, содержит ли какая-либо часть пути номера высот
        if any(num in part for part in folder_parts for num in ALTITUDE_NUMBERS):
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
            altitude_cm = altitude * 100  # Переводим в см
            gsd_horizontal = (altitude_cm * SENSOR_WIDTH_MM) / (focal_length * image_width)
            gsd_vertical = (altitude_cm * SENSOR_HEIGHT_MM) / (focal_length * image_height)
            width_meters = (gsd_horizontal * image_width) / 100  # Переводим см в метры
            height_meters = (gsd_vertical * image_height) / 100
            resolution_cm_per_pixel = (gsd_horizontal + gsd_vertical) / 2
            size_in_pixels = 100 / resolution_cm_per_pixel  # Для объекта размером 1 метр

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
