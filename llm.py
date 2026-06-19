from ollama import chat


def answer_with_rag(question: str, context: str, model_name: str = "gemma4") -> str:
    system_prompt = """
You are a careful document QA assistant.

Answer ONLY from the provided context.
Do not use outside knowledge.
If the answer is not supported by the context, say:
"I couldn't find that in the uploaded PDFs."

When the user asks for a comparison:
- compare across documents clearly
- group findings by document when useful
- identify similarities and differences
- mention the file names and page numbers that support each claim

Prefer this structure:
1. Direct answer
2. Comparison / breakdown
3. Sources

Do not invent citations.
"""

    user_prompt = f"""
Question:
{question}

Context:
{context}
"""

    response = chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
    )

    return response["message"]["content"]