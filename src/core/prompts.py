"""Prompt building utilities for the chatbot."""
from src.config import GREETING_PATTERN, DIRECT_CONTENT_HINTS, MAX_HISTORY_MESSAGES, normalize_question

def format_history(chat_history: list[dict[str, str]]) -> str:
    """
    Format chat history for inclusion in prompts.
    
    Args:
        chat_history: List of message dictionaries with 'role' and 'content'
        
    Returns:
        Formatted conversation string
    """
    if not chat_history:
        return "No prior conversation."

    lines = []
    for message in chat_history[-MAX_HISTORY_MESSAGES:]:
        role = message["role"].capitalize()
        lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)

def classify_query(question: str) -> str:
    """
    Classify a query into one of three categories:
    - small_talk: Greeting or casual message
    - direct_context: User provides content or asks for structured output
    - retrieval: Standard question requiring knowledge base lookup
    
    Args:
        question: User question
        
    Returns:
        Query classification string
    """
    normalized_question = normalize_question(question)
    if GREETING_PATTERN.match(normalized_question):
        return "small_talk"
    if any(hint in normalized_question for hint in DIRECT_CONTENT_HINTS):
        return "direct_context"
    if len(question) > 1200:
        return "direct_context"
    return "retrieval"

def build_prompt(question: str, context: str, history: str) -> str:
    """
    Build the main Q&A prompt for the LLM.
    
    Args:
        question: User question
        context: Retrieved context from knowledge base
        history: Formatted chat history
        
    Returns:
        Complete prompt for the LLM
    """
    return f"""You are a helpful chatbot for the Akshaya Goshala knowledge base.
Use the recent conversation to understand follow-up questions.
Answer only from the provided context when giving factual information.
Keep the answer short, direct, and natural.
If the answer is not in the context, say: Sorry,I missed something.
Conversation:
{history}
Context:
{context}
Question:
{question}
Answer:"""

def build_direct_context_prompt(question: str, history: str) -> str:
    """
    Build a specialized prompt for direct context or structured output requests.
    
    Args:
        question: User question
        history: Formatted chat history
        
    Returns:
        Complete prompt for the LLM
    """
    return f"""You are a careful assistant.
The user may provide full source content directly inside the request.
When that happens, answer from the user-provided content instead of external knowledge.
If the user asks for JSON, return valid JSON only.
Conversation:
{history}
User request:
{question}
Answer:"""
