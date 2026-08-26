import logging
import os
from groq import Groq
from llm.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT

logger = logging.getLogger(__name__)
MODEL_NAME = "qwen/qwen3.6-27b"

def generate_answer(
    question,
    retrieved_chunks,
    model_name=MODEL_NAME,
):
    """
    Generate an answer using retrieved RAG context and Groq.

    Args:
        question (str):
            User's question.

        retrieved_chunks (list):
            Retrieved chunks returned by the vector database.

        model_name (str):
            Groq model used for answer generation.

    Returns:
        dict:
            Structured LLM generation result.
    """

    logger.info("Entering Groq LLM generation method")

    # ---------------------------------------------------------
    # 1. Validate question
    # ---------------------------------------------------------

    if not question or not question.strip():
        logger.warning("Empty question received")

        return {
            "success": False,
            "question": question,
            "answer": None,
            "model_name": model_name,
            "error": {
                "type": "EMPTY_QUESTION",
                "message": "A non-empty question is required.",
            },
        }

    # ---------------------------------------------------------
    # 2. Validate retrieved chunks
    # ---------------------------------------------------------

    if not retrieved_chunks:
        logger.warning(
            "No retrieved chunks were provided for LLM generation"
        )

        return {
            "success": False,
            "question": question,
            "answer": None,
            "model_name": model_name,
            "error": {
                "type": "EMPTY_CONTEXT",
                "message": (
                    "No retrieved chunks were provided. "
                    "The LLM cannot answer from the RAG context."
                ),
            },
        }

    # ---------------------------------------------------------
    # 3. Validate Groq API key
    # ---------------------------------------------------------

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY is not configured")
        return {
            "success": False,
            "question": question,
            "answer": None,
            "model_name": model_name,
            "error": {
                "type": "GROQ_API_KEY_MISSING",
                "message": "GROQ_API_KEY is not configured.",
            },
        }

    # ---------------------------------------------------------
    # 4. Prepare retrieved context
    # ---------------------------------------------------------

    try:
        logger.info(
            "Preparing %s retrieved chunks for LLM context",
            len(retrieved_chunks),
        )
        context_parts = []
        for index, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            page_content = chunk.get(
                "page_content",
                "",
            )
            metadata = chunk.get(
                "metadata",
                {},
            )
            score = chunk.get(
                "score"
            )
            if not page_content:
                logger.warning(
                    "Retrieved chunk %s contains no page content",
                    index,
                )
                continue
            context_parts.append(
                f"""
--- Retrieved Context {index} ---
Similarity Score: {score}
Metadata: {metadata}

{page_content}
""".strip()
            )

        if not context_parts:
            logger.warning(
                "Retrieved chunks contained no usable content"
            )

            return {
                "success": False,
                "question": question,
                "answer": None,
                "model_name": model_name,
                "error": {
                    "type": "EMPTY_CONTEXT_CONTENT",
                    "message": (
                        "Retrieved chunks were provided, "
                        "but none contained usable content."
                    ),
                },
            }
        context = "\n\n".join(
            context_parts
        )
        logger.info(
            "Successfully prepared %s context chunks",
            len(context_parts),
        )
    except Exception as error:
        logger.exception(
            "Failed to prepare retrieved context"
        )
        return {
            "success": False,
            "question": question,
            "answer": None,
            "model_name": model_name,
            "error": {
                "type": "CONTEXT_PREPARATION_ERROR",
                "message": str(error),
            },
        }
    # ---------------------------------------------------------
    # 5. Initialize Groq client
    # ---------------------------------------------------------

    try:
        logger.info(
            "Initializing Groq client"
        )
        client = Groq(
            api_key=groq_api_key,
        )
    except Exception as error:
        logger.exception(
            "Failed to initialize Groq client"
        )
        return {
            "success": False,
            "question": question,
            "answer": None,
            "model_name": model_name,
            "error": {
                "type": "GROQ_CLIENT_ERROR",
                "message": str(error),
            },
        }

    # ---------------------------------------------------------
    # 6. Build prompts
    # ---------------------------------------------------------

    system_prompt = RAG_SYSTEM_PROMPT
    user_prompt = RAG_USER_PROMPT.format(
        context=context,
        question=question,
    )

    # ---------------------------------------------------------
    # 7. Generate answer using Groq
    # ---------------------------------------------------------

    try:
        logger.info(
            "Sending request to Groq model: %s",
            model_name,
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
        )

        # -----------------------------------------------------
        # 8. Extract answer
        # -----------------------------------------------------

        answer = response.choices[0].message.content
        if not answer or not answer.strip():
            logger.warning(
                "Groq returned an empty response"
            )
            return {
                "success": False,
                "question": question,
                "answer": None,
                "model_name": model_name,
                "error": {
                    "type": "EMPTY_LLM_RESPONSE",
                    "message": (
                        "Groq returned an empty response."
                    ),
                },
            }
        logger.info(
            "Groq LLM generation completed successfully"
        )

        # -----------------------------------------------------
        # 9. Return successful result
        # -----------------------------------------------------

        return {
            "success": True,
            "question": question,
            "answer": answer.strip(),
            "model_name": model_name,
            "retrieved_chunks": len(context_parts),
            "error": None,
        }

    except Exception as error:
        logger.exception(
            "Groq LLM generation failed"
        )
        return {
            "success": False,
            "question": question,
            "answer": None,
            "model_name": model_name,
            "error": {
                "type": "GROQ_GENERATION_ERROR",
                "message": str(error),
            },
        }