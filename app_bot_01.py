from tqdm import tqdm
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import re
import codecs
import time


def load_text(file_path):
    """Загрузка текста из файла."""
    with codecs.open(file_path, "r", encoding="utf-8", errors="ignore") as input_file:
        return input_file.read()


def clean_text(text):
    """
    Очищаем текст от лишних символов, чтобы подготовить его к обработке.
    """
    text = re.sub(r"\*\*.*?\*\*", "", text)  # Убираем заголовки
    text = re.sub(r"[\n\-]+", " ", text)  # Убираем переносы строк и дефисы
    text = re.sub(r"\s+", " ", text).strip()  # Убираем лишние пробелы
    return text


def generate_answer(question, context):
    """
    Генерация ответа с помощью модели GPT с отображением прогресса.
    """
    # Используем доступную бесплатную модель
    model_name = "ai-forever/ruGPT-3.5-13B"
    print(f"[INFO] Загружаем модель: {model_name}")

    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f"[INFO] Модель загружена за {time.time() - start_time:.2f} секунд.")

    gpt_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

    # Формируем подсказку для модели
    prompt = f"Контекст: {context}\n\nВопрос: {question}\nОтвет:"

    print(f"[INFO] Начинаем генерацию...")
    response = gpt_pipeline(prompt, max_length=300, num_return_sequences=1)

    print(f"[INFO] Генерация завершена за {time.time() - start_time:.2f} секунд.")
    # Извлекаем текст ответа
    return response[0]['generated_text'].split("Ответ:")[1].strip()


if __name__ == '__main__':
    # Загружаем базу знаний
    knowledge_base = load_text('OrderDeliciousBot_KnowledgeBase_01.txt')

    # Очищаем базу знаний
    cleaned_knowledge_base = clean_text(knowledge_base)

    # Пример пользовательского вопроса
    user_question = "Какие произведения написал Александр Пушкин?"

    # Генерация ответа
    start_time = time.time()
    answer = generate_answer(user_question, cleaned_knowledge_base)
    print(f"[INFO] Ответ сгенерирован за {time.time() - start_time:.2f} секунд.")

    print(f"Вопрос: {user_question}")
    print(f"Ответ: {answer}")
