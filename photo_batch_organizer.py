import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Параметры
SOURCE_DIR = r'C:\Users\mkolp\OneDrive\Документы\Данные для разметки\20241018'  # Исходная папка с фотографиями
DESTINATION_BASE_DIR = r'C:\Users\mkolp\OneDrive\Документы\Батчи'  # Общая папка для сохранения выборок
N = 20  # Количество фотографий в каждой выборке
MAX_WORKERS = 4  # Количество потоков для параллельной обработки
SUMMARY_FILE_PATH = os.path.join(DESTINATION_BASE_DIR, "summary.txt")  # Общий файл с описанием выборок

# Функция для обработки одного батча фотографий
def process_batch(batch_index, batch_files):
    # Создаем папку для текущей выборки
    batch_folder_name = f'batch_{batch_index + 1}'  # Номер группы
    batch_folder_path = os.path.join(DESTINATION_BASE_DIR, batch_folder_name)
    os.makedirs(batch_folder_path, exist_ok=True)

    # Создаем файл с именами фотографий в текущей выборке
    batch_txt_path = os.path.join(batch_folder_path, "photos_in_batch.txt")
    with open(batch_txt_path, 'w', encoding='utf-8') as batch_file:
        for photo in batch_files:
            batch_file.write(f"{photo}\n")

    # Копируем файлы в подкаталог
    for photo in batch_files:
        source_path = os.path.join(SOURCE_DIR, photo)
        destination_path = os.path.join(batch_folder_path, photo)
        shutil.copy2(source_path, destination_path)

    return batch_folder_name, batch_files

def main():
    # Проверка и создание общей папки
    if not os.path.exists(DESTINATION_BASE_DIR):
        os.makedirs(DESTINATION_BASE_DIR)

    # Получаем список всех фотографий из исходной папки
    photo_files = [
        f for f in os.listdir(SOURCE_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif'))
    ]


    # Перемешиваем список для случайной выборки, если требуется
    random.shuffle(photo_files)

    # Параллельная обработка батчей
    with open(SUMMARY_FILE_PATH, 'w', encoding='utf-8') as summary_file, ThreadPoolExecutor(
            max_workers=MAX_WORKERS) as executor:
        summary_file.write("Список фотографий в каждой выборке:\n\n")
        futures = []

        # Создаем задачи для каждого батча
        for i in range(0, len(photo_files), N):
            batch_files = photo_files[i:i + N]
            batch_index = i // N
            futures.append(executor.submit(process_batch, batch_index, batch_files))

        # Ожидаем завершения всех задач
        for future in as_completed(futures):
            batch_folder_name, batch_files = future.result()

            # Записываем информацию о текущей выборке в общий файл
            summary_file.write(f"{batch_folder_name}:\n")
            for photo in batch_files:
                summary_file.write(f"  {photo}\n")
            summary_file.write("\n")  # Добавляем пустую строку между группами для удобства

            print(f'Выборка {batch_folder_name} создана, содержит {len(batch_files)} фотографий.')

    print("Разделение и копирование фотографий завершено.")
    print(f"Общий файл с описанием выборок сохранен по пути: {SUMMARY_FILE_PATH}")

if __name__ == '__main__':
    main()