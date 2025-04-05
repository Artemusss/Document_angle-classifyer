import os
import time
from PIL import Image
import pytesseract

# Функция предсказания ориентации одного изображения
def predict(image_path):
    image = Image.open(image_path)
    
    # Получаем данные OSD (Orientation and Script Detection)
    osd_data = pytesseract.image_to_osd(image)
    
    # Извлекаем угол поворота
    angle = int(osd_data.split("Rotate:")[1].split("\n")[0].strip())

    print(f'Image: {image_path}')
    print(f'Predicted angle: {angle}°')
    
    return angle

# Функция тестирования на папке с изображениями
def test_model(test_dir):
    wrong_predictions = []  # Список ошибок
    sum_time = 0
    total_images = 0

    for class_folder in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_folder)

        if not os.path.isdir(class_path):
            continue  # Пропускаем файлы, если они вдруг есть

        true_angle = int(class_folder)  # Настоящий угол (название папки)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            start = time.time()
            predicted_angle = predict(img_path)  # Определяем угол
            end = time.time()
            
            sum_time += end - start
            total_images += 1

            if predicted_angle != true_angle:
                wrong_predictions.append({
                    'path': img_path,
                    'predicted': predicted_angle,
                    'real': true_angle
                })

    accuracy = 100 * (total_images - len(wrong_predictions)) / total_images if total_images > 0 else 0

    print(f"\nTest accuracy: {accuracy:.2f}%")
    print(f"Total time spent: {sum_time:.2f} sec")

    print(f"Wrong predictions ({len(wrong_predictions)}):")
    for item in wrong_predictions:
        print(f"Image: {item['path']}, Predicted: {item['predicted']}, Real: {item['real']}\n")

# Запуск тестирования
test_model('Test_images/test')
