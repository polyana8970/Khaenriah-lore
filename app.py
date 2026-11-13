from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import json

# ВАЖНО: здесь два подчёркивания с каждой стороны!
app = Flask(__name__)

# Загружаем модель для эмбеддингов (мультиязычная)
print("Загрузка модели эмбеддингов...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Загружаем текст
print("Загрузка данных...")
with open('data/lore.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Разбиваем текст на чанки по абзацам
chunks = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
print(f"Найдено {len(chunks)} фрагментов")

# Создаём эмбеддинги для каждого чанка
print("Создание эмбеддингов...")
embeddings = model.encode(chunks)

# Создаём FAISS индекс
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

# Функция поиска похожих чанков
def search_chunks(query, k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
    results = []
    for i in indices[0]:
        if i < len(chunks):
            results.append(chunks[i])
    return results

# Функция для получения ответа от Ollama (Qwen)
def get_answer(question, context):
    prompt = f"""Используй только эту информацию для ответа на вопрос. Отвечай на русском языке. Если ответа нет в тексте, так и скажи.

Контекст:
{context}

Вопрос: {question}

Ответ:"""
    
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'qwen2:7b',
                'prompt': prompt,
                'stream': False
            }
        )
        result = response.json()
        return result.get('response', 'Не удалось получить ответ')
    except Exception as e:
        return f"Ошибка подключения к Ollama: {e}. Убедись, что Ollama запущена."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question')
    
    if not question:
        return jsonify({'error': 'Введите вопрос'}), 400
    
    try:
        # Ищем похожие чанки
        chunks_found = search_chunks(question, k=5)
        context = '\n\n'.join(chunks_found)
        
        # Получаем ответ от LLM
        answer = get_answer(question, context)
        
        return jsonify({
            'answer': answer,
            'sources': chunks_found[:2]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ВАЖНО: здесь тоже два подчёркивания с каждой стороны!
if __name__ == '__main__':
    app.run(debug=True, port=5000)