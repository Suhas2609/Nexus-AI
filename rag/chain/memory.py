from langchain_classic.memory import ConversationBufferWindowMemory
from config.settings import settings


def create_memory():
    return ConversationBufferWindowMemory(
        k = settings.memory_window_k,
        memory_key = "chat_history",
        return_messages = True,
        output_key = "answer",
        input_key="input",
    )