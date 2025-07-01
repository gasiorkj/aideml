"""Backend for OpenRouter API"""

import json
import logging
import os
import time

from funcy import notnone, once, select_values
import openai
from openai import OpenAI

from .utils import FunctionSpec, OutputType, backoff_create, opt_messages_to_list

logger = logging.getLogger("aide")



_client: openai.OpenAI = None  # type: ignore

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

@once
def _setup_openrouter_client():
    global _client
    _client = openai.OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        default_headers={'OPENAI_API_KEY': os.getenv("OPENAI_API_KEY")},
        max_retries=0,
    )

    _client = OpenAI(
    api_key=os.getenv("RITS_API_KEY"),
    base_url=os.getenv("RITS_BASE_URL"),
    default_headers={'RITS_API_KEY': os.getenv("RITS_API_KEY")}
    )


# def query(
#     system_message: str | None,
#     user_message: str | None,
#     func_spec: FunctionSpec | None = None,
#     **model_kwargs,
# ) -> tuple[OutputType, float, int, int, dict]:
#     _setup_openrouter_client()
#     filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

#     if func_spec is not None:
#         tools = [func_spec.as_openai_tool_dict]
#         tool_choice = func_spec.openai_tool_choice_dict

#     # in case some backends dont support system roles, just convert everything to user
#     messages = [
#         {"role": "user", "content": message}
#         for message in [system_message, user_message]
#         if message
#     ]

#     t0 = time.time()
#     completion = backoff_create(
#         _client.chat.completions.create,
#         OPENAI_TIMEOUT_EXCEPTIONS,
#         messages=messages,
#         tools=tools,
#         tool_choice=tool_choice,
#         extra_body={
#             "provider": {
#                 "order": ["Fireworks"],
#                 "ignore": ["Together", "DeepInfra", "Hyperbolic"],
#             },
#         },
#         **filtered_kwargs,
#     )
#     req_time = time.time() - t0

#     output = completion.choices[0].message.content

#     in_tokens = completion.usage.prompt_tokens
#     out_tokens = completion.usage.completion_tokens

#     info = {
#         "system_fingerprint": completion.system_fingerprint,
#         "model": completion.model,
#         "created": completion.created,
#     }

#     return output, req_time, in_tokens, out_tokens, info


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query the OpenAI API, optionally with function calling.
    If the model doesn't support function calling, gracefully degrade to text generation.
    """
    _setup_openrouter_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)

    # Convert system/user messages to the format required by the client
    messages = opt_messages_to_list(system_message, user_message)

    # If function calling is requested, attach the function spec
    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    completion = None
    t0 = time.time()

    # Attempt the API call
    try:
        completion = backoff_create(
            _client.chat.completions.create,
            OPENAI_TIMEOUT_EXCEPTIONS,
            messages=messages,
            **filtered_kwargs,
        )
    except openai.BadRequestError as e:
        # Check whether the error indicates that function calling is not supported
        if "function calling" in str(e).lower() or "tools" in str(e).lower():
            logger.warning(
                "Function calling was attempted but is not supported by this model. "
                "Falling back to plain text generation."
            )
            # Remove function-calling parameters and retry
            filtered_kwargs.pop("tools", None)
            filtered_kwargs.pop("tool_choice", None)

            # Retry without function calling
            completion = backoff_create(
                _client.chat.completions.create,
                OPENAI_TIMEOUT_EXCEPTIONS,
                messages=messages,
                **filtered_kwargs,
            )
        else:
            # If it's some other error, re-raise
            raise

    req_time = time.time() - t0
    choice = completion.choices[0]

    # Decide how to parse the response
    if func_spec is None or "tools" not in filtered_kwargs:
        # No function calling was ultimately used
        output = choice.message.content
    else:
        # Attempt to extract tool calls
        tool_calls = getattr(choice.message, "tool_calls", None)
        if not tool_calls:
            logger.warning(
                "No function call was used despite function spec. Fallback to text.\n"
                f"Message content: {choice.message.content}"
            )
            output = choice.message.content
        else:
            first_call = tool_calls[0]
            # Optional: verify that the function name matches
            if first_call.function.name != func_spec.name:
                logger.warning(
                    f"Function name mismatch: expected {func_spec.name}, "
                    f"got {first_call.function.name}. Fallback to text."
                )
                output = choice.message.content
            else:
                try:
                    output = json.loads(first_call.function.arguments)
                except json.JSONDecodeError as ex:
                    logger.error(
                        "Error decoding function arguments:\n"
                        f"{first_call.function.arguments}"
                    )
                    raise ex

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
