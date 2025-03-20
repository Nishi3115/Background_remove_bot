import os
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.types import FSInputFile
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from rembg import remove
from PIL import Image, ImageEnhance
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# Токен бота
TOKEN = "7760978837:AAGmuO8anfJ2egVrdXxhCmv5MmcHbE86pH8"  # Вставьте ваш токен

# Создаем экземпляр бота и диспетчера
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# Папка для хранения фото
BASE_DIR = "photos"
os.makedirs(BASE_DIR, exist_ok=True)

# Состояния для FSM
class PhotoState(StatesGroup):
    waiting_for_background = State()
    waiting_for_retry = State()

# Варианты фонов
BACKGROUND_OPTIONS = {
    "forest": "templates/forest.jpeg",
    "office": "templates/office.jpeg",
    "beach": "templates/beach.jpeg",
    "city": "templates/city.jpeg",
    "space": "templates/space.jpeg",
    "studio": "templates/studio.jpeg",
    "mountains": "templates/mountains.jpeg"
}

# Инициализация модели RMBG-2.0 (выполняется один раз при запуске бота)
rmbg_model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
torch.set_float32_matmul_precision('high')
rmbg_model.to('cpu')  # Используем CPU
rmbg_model.eval()  # Режим оценки

# Настройки для обработки изображений в RMBG-2.0
image_size = (1024, 1024)  # Размер, требуемый моделью
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@dp.message(Command("start"))
async def start_handler(message: Message):
    user_id = str(message.from_user.id)
    user_folder = os.path.join(BASE_DIR, user_id)

    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    
    await message.answer("Папка для хранения фото создана!")

@dp.message(lambda message: message.photo)
async def photo_handler(message: Message, state: FSMContext):
    user_id = str(message.from_user.id)
    user_folder = os.path.join(BASE_DIR, user_id)

    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    # Определяем следующий номер файла
    existing_files = [f for f in os.listdir(user_folder) if f.endswith(".jpg")]
    if existing_files:
        numbers = [int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit()]
        next_number = max(numbers, default=99) + 1
    else:
        next_number = 100

    # Получаем лучшее качество фото
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    destination = os.path.join(user_folder, f"{next_number}.jpg")

    # Загружаем фото
    await bot.download_file(file.file_path, destination)

    # Сохраняем данные в состоянии
    await state.update_data(photo_path=destination, next_number=next_number, used_rmbg=False)

    # Создаем кнопки для выбора фона
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Лес", callback_data="forest")],
        [InlineKeyboardButton(text="Офис", callback_data="office")],
        [InlineKeyboardButton(text="Пляж", callback_data="beach")],
        [InlineKeyboardButton(text="Город", callback_data="city")],
        [InlineKeyboardButton(text="Космос", callback_data="space")],
        [InlineKeyboardButton(text="Студия", callback_data="studio")],
        [InlineKeyboardButton(text="Горы", callback_data="mountains")]
    ])
    
    await message.answer("Выберите фон для фото:", reply_markup=keyboard)
    await state.set_state(PhotoState.waiting_for_background)

@dp.callback_query(PhotoState.waiting_for_background)
async def process_background_choice(callback: types.CallbackQuery, state: FSMContext):
    user_id = str(callback.from_user.id)
    user_folder = os.path.join(BASE_DIR, user_id)
    
    # Получаем данные из состояния
    data = await state.get_data()
    original_path = data["photo_path"]
    next_number = data["next_number"]
    used_rmbg = data.get("used_rmbg", False)  # Проверяем, использовалась ли RMBG-2.0
    processed_path = os.path.join(user_folder, f"{next_number}_processed.png")

    # Выбранный фон
    background_key = callback.data
    template_path = BACKGROUND_OPTIONS.get(background_key, "templates/studio.png")  # Studio как запасной

    # Загрузка изображения
    input_image = Image.open(original_path).convert("RGB")  # Убедимся, что изображение в RGB

    # Выбираем модель для удаления фона
    if used_rmbg:
        # Используем RMBG-2.0
        enhancer = ImageEnhance.Brightness(input_image)
        preprocessed_image = enhancer.enhance(1.3)  # Увеличиваем яркость

        # Подготовка изображения для модели
        input_images = transform_image(preprocessed_image).unsqueeze(0).to('cpu')

        # Предсказание маски с помощью RMBG-2.0
        with torch.no_grad():
            preds = rmbg_model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()  # Получаем маску
        pred_pil = transforms.ToPILImage()(pred)  # Преобразуем в PIL-изображение
        mask = pred_pil.resize(input_image.size, Image.Resampling.LANCZOS)  # Возвращаем исходный размер

        # Применяем маску к исходному изображению
        output_image = input_image.convert("RGBA")
        output_image.putalpha(mask)  # Добавляем альфа-канал
    else:
        # Используем Rembg (по умолчанию)
        output_image = remove(input_image)  # Удаляем фон

    # Проверяем наличие шаблона фона
    width, height = output_image.size
    if os.path.exists(template_path):
        background = Image.open(template_path).convert("RGBA")
        background = background.resize((width, height), Image.Resampling.LANCZOS)
    else:
        # Если шаблона нет, используем однотонный белый фон
        background = Image.new("RGBA", (width, height), (255, 255, 255, 255))

    # Создаем итоговое изображение
    result_image = Image.new("RGBA", (width, height))
    result_image.paste(background, (0, 0))  # Сначала фон
    result_image.paste(output_image, (0, 0), output_image)  # Накладываем объект

    # Сохраняем результат
    result_image.save(processed_path, "PNG")

    # Создаем кнопку "Использовать другую модель"
    retry_keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Использовать другую модель", callback_data=f"retry_{background_key}")]
    ])

    # Отправляем обработанное фото с кнопкой
    photo_file = FSInputFile(processed_path)
    await bot.send_photo(
        callback.message.chat.id,
        photo_file,
        caption=f"Фото с фоном: {background_key.capitalize()}",
        reply_markup=retry_keyboard
    )

    # Сохраняем данные для возможного повторного запроса
    await state.update_data(background_key=background_key)
    await state.set_state(PhotoState.waiting_for_retry)
    await callback.answer()

@dp.callback_query(lambda c: c.data.startswith("retry_"), PhotoState.waiting_for_retry)
async def retry_background_removal(callback: types.CallbackQuery, state: FSMContext):
    # Получаем данные из состояния
    data = await state.get_data()
    background_key = data["background_key"]

    # Обновляем флаг, чтобы использовать RMBG-2.0
    await state.update_data(used_rmbg=True)

    # Создаем новый объект CallbackQuery с нужным background_key
    new_callback = types.CallbackQuery(
        id=callback.id,
        from_user=callback.from_user,
        chat_instance=callback.chat_instance,
        message=callback.message,
        data=background_key  # Устанавливаем нужный фон
    )

    # Повторно вызываем обработку с той же фоновой картинкой
    await process_background_choice(new_callback, state)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())