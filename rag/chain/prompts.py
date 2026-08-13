from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

CONTEXTUALISE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given a chat history and the latest user question, "
        "rewrite the question as a standalone query that can be understood "
        "without the chat history. Do not answer the question; only reformulate it if needed. "
        "Otherwise return it unchanged.",
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the user's question strictly using the "
        "information provided in the <context>. If the answer is not present in the "
        "context, say \"I don't have enough information in the provided documents "
        "to answer that.\" Do not use outside knowledge.",
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    (
        "human",
        "Context:\n{context}\n\nQuestion: {input}",
    ),
])
