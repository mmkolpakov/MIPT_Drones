import os
import re

def replace_class_in_txt_files(directory_path):
    """
    Обходит все папки и файлы по заданному пути, заменяя класс объекта в txt файлах на 0.
    """
    # Регулярное выражение для проверки строки формата: "число число число число число"
    pattern = re.compile(r'^\d+(\s\d+\.\d+){4}$')

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)

                try:
                    # Чтение содержимого файла
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    # Обработка строк
                    updated_lines = []
                    for line in lines:
                        line = line.strip()
                        if pattern.match(line):  # Проверка строки на соответствие шаблону
                            parts = line.split()
                            parts[0] = '0'  # Заменяем класс на 0
                            updated_lines.append(' '.join(parts) + '\n')
                        else:
                            updated_lines.append(line + '\n')  # Оставляем строку без изменений

                    # Запись обновленных строк обратно в файл
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(updated_lines)

                    print(f'Обработан файл: {file_path}')

                except Exception as e:
                    print(f'Ошибка при обработке файла {file_path}: {e}')


# Укажите путь к директории
directory_path = r'C:\Users\mkolp\OneDrive\Документы\5_course\Батчи_разметка\Батчи'

replace_class_in_txt_files(directory_path)

print('Обработка завершена.')
