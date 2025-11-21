import logging
import os
import json
import asyncio
from pathlib import Path
from typing import Any, List, Dict

from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from openai import OpenAI


# ------------------- CONFIG -------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8548243687:AAHdWvt9NUznUuInXrwxeNXp8N1uh_0fYLg")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-svowdqWQ7105Kr2eDf3515Ff0861403083EbA3F4Fb2b0f3a")
LAOZHANG_BASE_URL = os.getenv("LAOZHANG_BASE_URL", "https://api.laozhang.ai/v1")

# История чатов в виде json 
HISTORIES_DIR = Path("./histories")
HISTORIES_DIR.mkdir(exist_ok=True)

# Keep max entries in context to avoid exceeding model limits
MAX_CONTEXT_MESSAGES = 40

client = None
if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY":
    try:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=LAOZHANG_BASE_URL)
    except Exception:
        client = None

# In-memory cache of user contexts (chat histories)
user_contexts: Dict[int, List[Dict[str, str]]] = {}

# ------------------- History utilities -------------------

def history_path(chat_id: int) -> Path:
    return HISTORIES_DIR / f"{chat_id}.json"


def load_history(chat_id: int) -> List[Dict[str, str]]:
    path = history_path(chat_id)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def save_history(chat_id: int, messages: List[Dict[str, str]]):
    path = history_path(chat_id)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Не удалось сохранить историю для {chat_id}: {e}")


def reset_history(chat_id: int):
    user_contexts[chat_id] = []
    path = history_path(chat_id)
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def ensure_system_prompt(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not messages or messages[0].get("role") != "system":
        system_msg = {
            "role": "system",
            "content": "Ты — полезный и вежливый ассистент. Отвечай на русском языке, если пользователь не попросит иначе.",
        }
        return [system_msg] + messages
    return messages


def trim_context(messages: List[Dict[str, str]], max_messages: int = MAX_CONTEXT_MESSAGES) -> List[Dict[str, str]]:
    if not messages:
        return messages
    system = messages[0] if messages and messages[0].get("role") == "system" else None
    body = messages[1:] if system else messages
    allowed = max_messages - (1 if system else 0)
    if len(body) > allowed:
        body = body[-allowed:]
    return ([system] if system else []) + body

# ------------------- Bot handlers -------------------
async def start(update: Any, context: Any):
    chat_id = getattr(update.effective_chat, "id", 0)
    reset_history(chat_id)

    keyboard = ReplyKeyboardMarkup([["Новый запрос"]], resize_keyboard=True)

    await update.message.reply_text(
        (
            "Привет! Я бот, который использует ChatGPT через LaoZhang API.\n"
            "Напиши любой текст — и я отвечу.\n\n"
            "Команды:\n"
            "/start — начать заново (очистить контекст)\n"
            "/help — подсказка"
        ),
        reply_markup=keyboard,
    )


async def help_handler(update: Any, context: Any):
    await update.message.reply_text(
        "Просто отправь любое сообщение — бот передаст его в модель и вернёт ответ.\nКоманда /start или кнопка 'Новый запрос' очищают контекст."
    )


async def new_request(update: Any, context: Any):
    chat_id = getattr(update.effective_chat, "id", 0)
    reset_history(chat_id)
    await update.message.reply_text("Контекст очищен. Введите новый запрос.")


async def call_model(messages: List[Dict[str, str]], model: str = "gpt-4o-mini", temperature: float = 0.7, max_tokens: int | None = None):
    if client is None:
        raise RuntimeError("OpenAI client not configured (OPENAI_API_KEY missing or library not installed)")

    def _call():
        kwargs: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return client.chat.completions.create(**kwargs)

    return await asyncio.to_thread(_call)


async def handle_message(update: Any, context: Any):
    chat_id = getattr(update.effective_chat, "id", 0)
    text = getattr(update.message, "text", "").strip()

    if text == "Новый запрос":
        return await new_request(update, context)

    if chat_id not in user_contexts:
        user_contexts[chat_id] = load_history(chat_id)

    user_contexts[chat_id].append({"role": "user", "content": text})

    messages = ensure_system_prompt(user_contexts[chat_id])
    messages = trim_context(messages)
    save_history(chat_id, messages)

    try:
        resp = await call_model(messages=messages, model="gpt-4o-mini", temperature=0.7, max_tokens=1500)
    except Exception as e:
        logging.exception("Ошибка при вызове модели:")
        await update.message.reply_text(f"Произошла ошибка при обращении к модели: {e}")
        return

    try:
        reply_text = resp.choices[0].message.content
    except Exception:
        logging.exception("Не удалось распарсить ответ модели:")
        await update.message.reply_text("Модель вернула некорректный ответ.")
        return

    user_contexts[chat_id] = trim_context(messages + [{"role": "assistant", "content": reply_text}])
    save_history(chat_id, user_contexts[chat_id])

    await update.message.reply_text(reply_text)

# ------------------- MAIN -------------------

def main():
    logging.basicConfig(level=logging.INFO)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
