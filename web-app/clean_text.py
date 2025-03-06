import re

def clean_text(text: str) -> str:
    """
    Очищает текст от HTML тегов, специальных символов и приводит его к нижнему регистру.
    Параметры:
    text (str): Исходный текст.
    Возвращает:
    str: Очищенный текст.
    """
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text
