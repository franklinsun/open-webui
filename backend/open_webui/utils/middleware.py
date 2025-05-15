import time
import logging
import sys
import os
import base64

import asyncio
from aiocache import cached
from typing import Any, Optional
import random
import json
import html
import inspect
import re
import ast

from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor


from fastapi import Request, HTTPException
from starlette.responses import Response, StreamingResponse


from open_webui.models.chats import Chats
from open_webui.models.users import Users
from open_webui.socket.main import (
    get_event_call,
    get_event_emitter,
    get_active_status_by_user_id,
)
from open_webui.routers.tasks import (
    generate_queries,
    generate_title,
    generate_image_prompt,
    generate_chat_tags,
)
from open_webui.routers.retrieval import process_web_search, SearchForm
from open_webui.routers.images import image_generations, GenerateImageForm
from open_webui.routers.pipelines import (
    process_pipeline_inlet_filter,
    process_pipeline_outlet_filter,
)

from open_webui.utils.webhook import post_webhook


from open_webui.models.users import UserModel
from open_webui.models.functions import Functions
from open_webui.models.models import Models

from open_webui.retrieval.utils import get_sources_from_files, get_sources_from_global_rag


from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.task import (
    get_task_model_id,
    rag_template,
    tools_function_calling_generation_template,
)
from open_webui.utils.misc import (
    deep_update,
    get_message_list,
    add_or_update_system_message,
    add_or_update_user_message,
    get_last_user_message,
    get_last_assistant_message,
    prepend_to_first_user_message_content,
    convert_logit_bias_input_to_json,
)
from open_webui.utils.tools import get_tools
from open_webui.utils.plugin import load_function_module_by_id
from open_webui.utils.filter import (
    get_sorted_filter_ids,
    process_filter_functions,
)
from open_webui.utils.code_interpreter import execute_code_jupyter

from open_webui.tasks import create_task

from open_webui.config import (
    CACHE_DIR,
    DEFAULT_TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE,
    DEFAULT_CODE_INTERPRETER_PROMPT,
)
from open_webui.env import (
    SRC_LOG_LEVELS,
    GLOBAL_LOG_LEVEL,
    BYPASS_MODEL_ACCESS_CONTROL,
    ENABLE_REALTIME_CHAT_SAVE,
)
from open_webui.constants import TASKS


logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])


async def chat_completion_tools_handler(
    request: Request, body: dict, extra_params: dict, user: UserModel, models, tools
) -> tuple[dict, dict]:
    """ 这个函数在非原生工具调用模式下 (metadata.get("function_calling") != "native") 被 process_chat_payload 调用。它的核心作用是：
        1. 使用一个特殊的提示模板 (TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE) 和工具规格 (tools_specs) 先调用一次 LLM (通常是 task_model_id 指定的模型)，让 LLM 判断应该调用哪个工具以及使用什么参数。
        2. 解析 LLM 的响应，获取工具调用指令。
        3. 遍历指令，查找 tools_dict 中对应的工具信息。
        4. 执行工具:
            如果工具是外部的 (direct: True)，它可能会使用 event_caller 发送 execute:tool 事件（委派执行）。
            如果工具是内部的 (direct: False)，直接调用 tool["callable"]。
        5.将工具执行结果添加到消息历史 (form_data["messages"]) 或 RAG 上下文 (sources) 中。
    Args:
        request (Request): FastAPI 请求对象
        body (dict): 包含聊天消息、模型 ID 等的字典 (类似 form_data)
        extra_params (dict): 包含事件发射器、用户信息、元数据等的字典
        user (UserModel): 当前用户信息
        models (_type_): 可用模型列表
        tools (_type_): 由 get_tools 准备好的工具字典 {tool_name: {"callable": ..., "spec": ..., "direct": ...}}

    Returns:
        tuple[dict, dict]: 一个元组 (更新后的 body, {"sources": sources})
    """
    # 定义一个异步辅助函数，用于从 LLM 响应中提取 'content' 字段。
    # 它处理流式和非流式响应。    
    async def get_content_from_response(response) -> Optional[str]:
        content = None
        if hasattr(response, "body_iterator"):
            # 如果是流式响应，异步迭代响应体
            async for chunk in response.body_iterator:
                # 解码块并解析 JSON
                data = json.loads(chunk.decode("utf-8"))
                # 提取消息内容
                content = data["choices"][0]["message"]["content"]

            # 清理后台任务（如果存在）
            if response.background is not None:
                await response.background()
        else:
            # 如果是非流式响应，直接提取内容
            content = response["choices"][0]["message"]["content"]
        return content
    # 定义一个辅助函数，用于构建发送给 LLM 以决定工具调用的特定 payload。
    def get_tools_function_calling_payload(messages, task_model_id, content):
        # 获取最后的用户消息
        user_message = get_last_user_message(messages)
        # 获取最近的4条消息作为历史记录
        history = "\n".join(
            f"{message['role'].upper()}: \"\"\"{message['content']}\"\"\""
            for message in messages[::-1][:4]
        )
        # 构建提示，包含历史记录和当前查询
        prompt = f"History:\n{history}\nQuery: {user_message}"
        # 返回构建好的 payload
        return {
            "model": task_model_id, # 使用指定的任务模型
            "messages": [
                {"role": "system", "content": content}, # 系统消息包含工具描述和指令
                {"role": "user", "content": f"Query: {prompt}"}, # 用户消息包含上下文和查询
            ],
            "stream": False, # 非流式，需要完整结果
            "metadata": {"task": str(TASKS.FUNCTION_CALLING)}, # 标记任务类型
        }
    # event_caller 用于通过 WebSocket 触发后端事件。
    # 在这里，它主要用于委派外部工具 (direct: True) 的执行。
    # 当需要调用外部工具时，会发送一个 execute:tool 事件，可能由另一个进程或任务来处理实际的 HTTP 请求。
    event_caller = extra_params["__event_call__"]
    # 包含当前请求的元数据，如 session_id，这对于 event_caller 路由事件可能很重要。
    metadata = extra_params["__metadata__"]

    task_model_id = get_task_model_id(
        body["model"],
        request.app.state.config.TASK_MODEL,
        request.app.state.config.TASK_MODEL_EXTERNAL,
        models,
    )
    # 这个标志用于控制后续的 RAG 文件处理。如果某个被调用的工具设置了 file_handler: True 元数据，
    # 这个标志会被设为 True，表示该工具已经处理了文件相关信息，
    # 后续的 RAG 文件处理步骤 (chat_completion_files_handler) 应该跳过用户上传的文件。
    skip_files = False
    sources = []

    specs = [tool["spec"] for tool in tools.values()]
    # 准备将工具规格注入到给 LLM 的提示中。
    tools_specs = json.dumps(specs)

    if request.app.state.config.TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE != "":
        template = request.app.state.config.TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE
    else:
        template = DEFAULT_TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE
    # 将序列化的工具规格 (tools_specs) 注入到选定的提示模板 (template) 中，生成最终的系统提示内容 (tools_function_calling_prompt)。这个提示会告诉 LLM 可用的工具以及如何选择它们。
    tools_function_calling_prompt = tools_function_calling_generation_template(
        template, tools_specs
    )
    # 使用聊天消息、选定的任务模型 ID 和生成的工具调用提示，创建用于第一次 LLM 调用的完整请求体 (payload)。
    payload = get_tools_function_calling_payload(
        body["messages"], task_model_id, tools_function_calling_prompt
    )
    log.debug(f"get_tools_function_calling_payload: {payload}")

    try:
        # 向指定的 task_model_id 发送请求，让它根据用户查询、聊天历史和提供的工具规格，决定应该调用哪个工具以及使用什么参数。
        response = await generate_chat_completion(request, form_data=payload, user=user)
        log.debug(f"{response=}")
        # 从 LLM 的响应中提取包含工具调用决策的文本内容。
        content = await get_content_from_response(response)
        log.debug(f"{content=}")
        # 如果 LLM 没有返回有效内容（例如，它决定不需要调用工具，或者调用失败），则直接返回原始的 body 和空的 flags，跳过后续的工具执行。
        if not content:
            return body, {}

        try:
            content = content[content.find("{") : content.rfind("}") + 1]
            if not content:
                raise Exception("No JSON object found in the response")

            result = json.loads(content)
            # 定义一个内部异步函数来处理单个工具调用指令
            async def tool_call_handler(tool_call):
                # 允许修改外部作用域的 skip_files 变量
                nonlocal skip_files

                log.debug(f"{tool_call=}")

                tool_function_name = tool_call.get("name", None)
                if tool_function_name not in tools:
                    return body, {}
                # 获取LLM建议的参数
                tool_function_params = tool_call.get("parameters", {})

                try:
                    tool = tools[tool_function_name]

                    spec = tool.get("spec", {})
                    allowed_params = (
                        spec.get("parameters", {}).get("properties", {}).keys()
                    )
                    # 过滤LLM提供的参数，只保留规格中允许的参数
                    tool_function_params = {
                        k: v
                        for k, v in tool_function_params.items()
                        if k in allowed_params
                    }
                    # 检查 'direct' 标志来区分内部工具和外部工具
                    if tool.get("direct", False):
                        tool_result = await event_caller(
                            {
                                "type": "execute:tool",
                                "data": {
                                    "id": str(uuid4()),
                                    "name": tool_function_name,
                                    "params": tool_function_params,
                                    "server": tool.get("server", {}),
                                    "session_id": metadata.get("session_id", None),
                                },
                            }
                        )
                    else:
                        tool_function = tool["callable"]
                        tool_result = await tool_function(**tool_function_params)

                except Exception as e:
                    tool_result = str(e)
                # --- 处理工具执行结果 ---
                tool_result_files = []
                if isinstance(tool_result, list):
                    for item in tool_result:
                        # check if string
                        if isinstance(item, str) and item.startswith("data:"):
                            tool_result_files.append(item)
                            tool_result.remove(item)

                if isinstance(tool_result, dict) or isinstance(tool_result, list):
                    tool_result = json.dumps(tool_result, indent=2)
                # --- 结果整合 ---
                if isinstance(tool_result, str):
                    tool = tools[tool_function_name]
                    tool_id = tool.get("tool_id", "")

                    tool_name = (
                        f"{tool_id}/{tool_function_name}"
                        if tool_id
                        else f"{tool_function_name}"
                    )
                    # 检查是否需要将结果作为 RAG 来源 (citation: True 或 外部工具)
                    if tool.get("metadata", {}).get("citation", False) or tool.get(
                        "direct", False
                    ):
                        # Citation is enabled for this tool
                        sources.append(
                            {
                                "source": {
                                    "name": (f"TOOL:{tool_name}"),
                                },
                                "document": [tool_result],
                                "metadata": [{"source": (f"TOOL:{tool_name}")}],
                            }
                        )
                    else:
                        # Citation is not enabled for this tool
                        body["messages"] = add_or_update_user_message(
                            f"\nTool `{tool_name}` Output: {tool_result}",
                            body["messages"],
                        )

                    if (
                        tools[tool_function_name]
                        .get("metadata", {})
                        .get("file_handler", False)
                    ):
                        skip_files = True

            # check if "tool_calls" in result
            if result.get("tool_calls"):
                for tool_call in result.get("tool_calls"):
                    await tool_call_handler(tool_call)
            else:
                await tool_call_handler(result)

        except Exception as e:
            log.debug(f"Error: {e}")
            content = None
    except Exception as e:
        log.debug(f"Error: {e}")
        content = None

    log.debug(f"tool_contexts: {sources}")

    if skip_files and "files" in body.get("metadata", {}):
        del body["metadata"]["files"]

    return body, {"sources": sources}


async def chat_web_search_handler(
    request: Request, form_data: dict, extra_params: dict, user
):
    """1. 调用 generate_queries (在 routers/tasks.py) 生成搜索查询。
        2. 调用 process_web_search (在 routers/retrieval.py) 执行 Web 搜索。
        3. 将搜索结果（文档内容或集合名称）添加到 form_data["files"] 中，以便后续 RAG 处理。
        4. 通过 event_emitter 向前端发送搜索状态更新。

    Args:
        request (Request): _description_
        form_data (dict): _description_
        extra_params (dict): _description_
        user (_type_): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    event_emitter = extra_params["__event_emitter__"]
    await event_emitter(
        {
            "type": "status",
            "data": {
                "action": "web_search",
                "description": "Generating search query",
                "done": False,
            },
        }
    )

    messages = form_data["messages"]
    user_message = get_last_user_message(messages)

    queries = []
    try:
        res = await generate_queries(
            request,
            {
                "model": form_data["model"],
                "messages": messages,
                "prompt": user_message,
                "type": "web_search",
            },
            user,
        )

        response = res["choices"][0]["message"]["content"]

        try:
            bracket_start = response.find("{")
            bracket_end = response.rfind("}") + 1

            if bracket_start == -1 or bracket_end == -1:
                raise Exception("No JSON object found in the response")

            response = response[bracket_start:bracket_end]
            queries = json.loads(response)
            queries = queries.get("queries", [])
        except Exception as e:
            queries = [response]

    except Exception as e:
        log.exception(e)
        queries = [user_message]

    if len(queries) == 0:
        await event_emitter(
            {
                "type": "status",
                "data": {
                    "action": "web_search",
                    "description": "No search query generated",
                    "done": True,
                },
            }
        )
        return form_data

    all_results = []

    for searchQuery in queries:
        await event_emitter(
            {
                "type": "status",
                "data": {
                    "action": "web_search",
                    "description": 'Searching "{{searchQuery}}"',
                    "query": searchQuery,
                    "done": False,
                },
            }
        )

        try:
            results = await process_web_search(
                request,
                SearchForm(
                    **{
                        "query": searchQuery,
                    }
                ),
                user=user,
            )

            if results:
                all_results.append(results)
                files = form_data.get("files", [])

                if results.get("collection_names"):
                    for col_idx, collection_name in enumerate(
                        results.get("collection_names")
                    ):
                        files.append(
                            {
                                "collection_name": collection_name,
                                "name": searchQuery,
                                "type": "web_search",
                                "urls": [results["filenames"][col_idx]],
                            }
                        )
                elif results.get("docs"):
                    # Invoked when bypass embedding and retrieval is set to True
                    docs = results["docs"]

                    if len(docs) == len(results["filenames"]):
                        # the number of docs and filenames (urls) should be the same
                        for doc_idx, doc in enumerate(docs):
                            files.append(
                                {
                                    "docs": [doc],
                                    "name": searchQuery,
                                    "type": "web_search",
                                    "urls": [results["filenames"][doc_idx]],
                                }
                            )
                    else:
                        # edge case when the number of docs and filenames (urls) are not the same
                        # this should not happen, but if it does, we will just append the docs
                        files.append(
                            {
                                "docs": results.get("docs", []),
                                "name": searchQuery,
                                "type": "web_search",
                                "urls": results["filenames"],
                            }
                        )

                form_data["files"] = files
        except Exception as e:
            log.exception(e)
            await event_emitter(
                {
                    "type": "status",
                    "data": {
                        "action": "web_search",
                        "description": 'Error searching "{{searchQuery}}"',
                        "query": searchQuery,
                        "done": True,
                        "error": True,
                    },
                }
            )

    if all_results:
        urls = []
        for results in all_results:
            if "filenames" in results:
                urls.extend(results["filenames"])

        await event_emitter(
            {
                "type": "status",
                "data": {
                    "action": "web_search",
                    "description": "Searched {{count}} sites",
                    "urls": urls,
                    "done": True,
                },
            }
        )
    else:
        await event_emitter(
            {
                "type": "status",
                "data": {
                    "action": "web_search",
                    "description": "No search results found",
                    "done": True,
                    "error": True,
                },
            }
        )

    return form_data


async def chat_image_generation_handler(
    request: Request, form_data: dict, extra_params: dict, user
):
    """1. 调用 generate_image_prompt (在 routers/tasks.py) 优化生成提示。
        2. 调用 image_generations (在 routers/images.py) 调用配置的图像生成引擎 API。
        3. 通过 event_emitter 向前端发送生成的图像文件信息和状态更新。
        4. 向 form_data["messages"] 添加一条系统消息，告知 LLM 图像已生成（或生成失败）。

    Args:
        request (Request): _description_
        form_data (dict): _description_
        extra_params (dict): _description_
        user (_type_): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    __event_emitter__ = extra_params["__event_emitter__"]
    await __event_emitter__(
        {
            "type": "status",
            "data": {"description": "Generating an image", "done": False},
        }
    )

    messages = form_data["messages"]
    user_message = get_last_user_message(messages)

    prompt = user_message
    negative_prompt = ""

    if request.app.state.config.ENABLE_IMAGE_PROMPT_GENERATION:
        try:
            res = await generate_image_prompt(
                request,
                {
                    "model": form_data["model"],
                    "messages": messages,
                },
                user,
            )

            response = res["choices"][0]["message"]["content"]

            try:
                bracket_start = response.find("{")
                bracket_end = response.rfind("}") + 1

                if bracket_start == -1 or bracket_end == -1:
                    raise Exception("No JSON object found in the response")

                response = response[bracket_start:bracket_end]
                response = json.loads(response)
                prompt = response.get("prompt", [])
            except Exception as e:
                prompt = user_message

        except Exception as e:
            log.exception(e)
            prompt = user_message

    system_message_content = ""

    try:
        images = await image_generations(
            request=request,
            form_data=GenerateImageForm(**{"prompt": prompt}),
            user=user,
        )

        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": "Generated an image", "done": True},
            }
        )

        await __event_emitter__(
            {
                "type": "files",
                "data": {
                    "files": [
                        {
                            "type": "image",
                            "url": image["url"],
                        }
                        for image in images
                    ]
                },
            }
        )

        system_message_content = "<context>User is shown the generated image, tell the user that the image has been generated</context>"
    except Exception as e:
        log.exception(e)
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": f"An error occurred while generating an image",
                    "done": True,
                },
            }
        )

        system_message_content = "<context>Unable to generate an image, tell the user that an error occurred</context>"

    if system_message_content:
        form_data["messages"] = add_or_update_system_message(
            system_message_content, form_data["messages"]
        )

    return form_data


async def chat_completion_files_handler(
    request: Request, body: dict, user: UserModel
) -> tuple[dict, dict[str, list]]:
    sources = []

    if files := body.get("metadata", {}).get("files", None):
        queries = []
        try:
            queries_response = await generate_queries(
                request,
                {
                    "model": body["model"],
                    "messages": body["messages"],
                    "type": "retrieval",
                },
                user,
            )
            queries_response = queries_response["choices"][0]["message"]["content"]

            try:
                bracket_start = queries_response.find("{")
                bracket_end = queries_response.rfind("}") + 1

                if bracket_start == -1 or bracket_end == -1:
                    raise Exception("No JSON object found in the response")

                queries_response = queries_response[bracket_start:bracket_end]
                queries_response = json.loads(queries_response)
            except Exception as e:
                queries_response = {"queries": [queries_response]}

            queries = queries_response.get("queries", [])
        except:
            pass

        if len(queries) == 0:
            queries = [get_last_user_message(body["messages"])]

        try:
            # Offload get_sources_from_files to a separate thread
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as executor:
                sources = await loop.run_in_executor(
                    executor,
                    lambda: get_sources_from_files(
                        request=request,
                        files=files,
                        queries=queries,
                        embedding_function=lambda query, prefix: request.app.state.EMBEDDING_FUNCTION(
                            query, prefix=prefix, user=user
                        ),
                        k=request.app.state.config.TOP_K,
                        reranking_function=request.app.state.rf,
                        k_reranker=request.app.state.config.TOP_K_RERANKER,
                        r=request.app.state.config.RELEVANCE_THRESHOLD,
                        hybrid_search=request.app.state.config.ENABLE_RAG_HYBRID_SEARCH,
                        full_context=request.app.state.config.RAG_FULL_CONTEXT,
                    ),
                )
        except Exception as e:
            log.exception(e)

        if request.app.state.config.USE_GLOBAL_RAG:
            collection_names = [
                name.strip() for name in request.app.state.config.COLLECTION_NAME.split(",")
            ]
            sources.extend(
                get_sources_from_global_rag(
                    collection_names=collection_names,
                    queries=queries,
                    embedding_function=lambda query: request.app.state.EMBEDDING_FUNCTION(
                        query, user=user
                    ),
                    k=request.app.state.config.TOP_K,
                    reranking_function=request.app.state.rf,
                    k_reranker=request.app.state.config.TOP_K_RERANKER,
                    r=request.app.state.config.RELEVANCE_THRESHOLD,
                    hybrid_search=request.app.state.config.ENABLE_RAG_HYBRID_SEARCH,
                    full_context=request.app.state.config.RAG_FULL_CONTEXT,
                )
            )

        log.debug(f"rag_contexts:sources: {sources}")

    return body, {"sources": sources}


def apply_params_to_form_data(form_data, model):
    params = form_data.pop("params", {})
    if model.get("ollama"):
        form_data["options"] = params

        if "format" in params:
            form_data["format"] = params["format"]

        if "keep_alive" in params:
            form_data["keep_alive"] = params["keep_alive"]
    else:
        if "seed" in params and params["seed"] is not None:
            form_data["seed"] = params["seed"]

        if "stop" in params and params["stop"] is not None:
            form_data["stop"] = params["stop"]

        if "temperature" in params and params["temperature"] is not None:
            form_data["temperature"] = params["temperature"]

        if "max_tokens" in params and params["max_tokens"] is not None:
            form_data["max_tokens"] = params["max_tokens"]

        if "top_p" in params and params["top_p"] is not None:
            form_data["top_p"] = params["top_p"]

        if "frequency_penalty" in params and params["frequency_penalty"] is not None:
            form_data["frequency_penalty"] = params["frequency_penalty"]

        if "reasoning_effort" in params and params["reasoning_effort"] is not None:
            form_data["reasoning_effort"] = params["reasoning_effort"]

        if "logit_bias" in params and params["logit_bias"] is not None:
            try:
                form_data["logit_bias"] = json.loads(
                    convert_logit_bias_input_to_json(params["logit_bias"])
                )
            except Exception as e:
                print(f"Error parsing logit_bias: {e}")

    return form_data


async def process_chat_payload(request, form_data, user, metadata, model):
    """负责处理各种前置任务，包括 RAG、Web 搜索、图像生成以及 工具调用。

    Args:
        request (_type_): 请求对象，用于访问应用状态和配置
        form_data (_type_): 包含聊天消息、模型 ID 等的字典
        user (_type_): 当前用户信息 (UserModel)
        metadata (_type_): 包含聊天 ID、消息 ID、会话 ID 等上下文信息的字典
        model (_type_): 当前选择的 LLM 模型信息字典

    Raises:
        e: _description_
        Exception: _description_
        Exception: _description_

    Returns:
        _type_: 一个元组 (form_data, metadata, events)
    """
    # 将 model 定义中包含的模型特定参数（例如温度 temperature、最大令牌数 max_tokens 等）合并到 form_data 中。
    # 这样可以确保发送给 LLM 的请求包含了模型预设的参数。
    form_data = apply_params_to_form_data(form_data, model)
    log.debug(f"form_data: {form_data}")
    # 获取用于 WebSocket 通信的函数。
    # event_emitter 用于向前端发送事件（例如状态更新），
    # event_call 用于从后端触发需要通过 WebSocket 处理的任务（例如执行代码解释器、调用某些工具）。
    # 它们通常与特定的会话 ID 和用户 ID 关联。
    event_emitter = get_event_emitter(metadata)
    event_call = get_event_call(metadata)
    # 这个字典捆绑了各种上下文信息（WebSocket 通信句柄、用户信息、请求元数据、FastAPI 请求对象、模型信息），
    # 以便后续传递给过滤器函数 (process_filter_functions) 和工具函数 (get_tools)。
    # 这允许这些函数访问当前的请求状态和用户信息。
    extra_params = {
        "__event_emitter__": event_emitter,
        "__event_call__": event_call,
        "__user__": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
        },
        "__metadata__": metadata,
        "__request__": request,
        "__model__": model,
    }

    # 检查请求状态中是否有 direct 标志（通常表示这是一个直接连接到外部 LLM API 的请求，而不是通过 Open WebUI 管理的模型）。
    # 如果是直接连接，则只使用该特定模型；否则，使用 request.app.state.MODELS 中缓存的所有可用模型列表（这些模型在 utils/models.py:get_all_models 中加载）
    if getattr(request.state, "direct", False) and hasattr(request.state, "model"):
        models = {
            request.state.model["id"]: request.state.model,
        }
    else:
        models = request.app.state.MODELS
    # 根据配置 (TASK_MODEL, TASK_MODEL_EXTERNAL) 和当前选择的模型 (form_data["model"])，
    # 确定用于执行内部后台任务（如标题生成、标签生成、RAG 查询生成等）的模型 ID。
    # 这允许为这些任务指定一个可能不同于主聊天模型的、更轻量或专门的模型。
    task_model_id = get_task_model_id(
        form_data["model"],
        request.app.state.config.TASK_MODEL,
        request.app.state.config.TASK_MODEL_EXTERNAL,
        models,
    )
    # events 用于存储需要在处理完成后发送给前端的事件（例如 RAG 检索到的来源信息）
    events = []
    # sources 用于累积从 RAG 或工具调用中获取的上下文来源信息。
    sources = []
    # 从 form_data["messages"] 列表中提取最后一条用户发送的消息内容。这通常用作 RAG 查询或 Web 搜索查询的基础。
    user_message = get_last_user_message(form_data["messages"])

    model_knowledge = model.get("info", {}).get("meta", {}).get("knowledge", False)
    if model_knowledge:
        # 通过 event_emitter 向前端发送一个状态更新事件，告知用户正在进行知识库搜索。
        await event_emitter(
            {
                "type": "status",
                "data": {
                    "action": "knowledge_search",
                    "query": user_message,
                    "done": False,
                },
            }
        )
        # 将模型配置中定义的知识库信息（集合名称等）添加到 form_data 的 files 列表中，以便后续 RAG 处理。
        knowledge_files = []
        for item in model_knowledge:
            if item.get("collection_name"):
                knowledge_files.append(
                    {
                        "id": item.get("collection_name"),
                        "name": item.get("name"),
                        "legacy": True,
                    }
                )
            elif item.get("collection_names"):
                knowledge_files.append(
                    {
                        "name": item.get("name"),
                        "type": "collection",
                        "collection_names": item.get("collection_names"),
                        "legacy": True,
                    }
                )
            else:
                knowledge_files.append(item)

        files = form_data.get("files", [])
        files.extend(knowledge_files)
        form_data["files"] = files

    variables = form_data.pop("variables", None)

    #  执行配置的“入口过滤器”(Inlet Filter)。
    # 这些过滤器是在请求发送给 LLM 之前 执行的自定义函数（通常是外部 Pipeline 服务），可以修改请求内容 (form_data)。
    try:
        form_data = await process_pipeline_inlet_filter(
            request, form_data, user, models
        )
    except Exception as e:
        raise e
    # 执行内部定义的过滤器函数（存储在数据库中，类型为 "filter"）。
    # 这些函数也可以在请求发送给 LLM 之前修改 form_data。extra_params 被传递给这些函数。
    try:
        filter_functions = [
            Functions.get_function_by_id(filter_id)
            for filter_id in get_sorted_filter_ids(model)
        ]

        form_data, flags = await process_filter_functions(
            request=request,
            filter_functions=filter_functions,
            filter_type="inlet",
            form_data=form_data,
            extra_params=extra_params,
        )
    except Exception as e:
        raise Exception(f"Error: {e}")
    # 检查用户是否在前端启用了特定功能。
    features = form_data.pop("features", None)
    if features:
        if "web_search" in features and features["web_search"]:
            form_data = await chat_web_search_handler(
                request, form_data, extra_params, user
            )

        if "image_generation" in features and features["image_generation"]:
            form_data = await chat_image_generation_handler(
                request, form_data, extra_params, user
            )

        # 将代码解释器的系统提示（来自配置 CODE_INTERPRETER_PROMPT_TEMPLATE 或默认值 DEFAULT_CODE_INTERPRETER_PROMPT）添加到 form_data["messages"] 中。
        # 这指示 LLM 可以使用代码解释器工具。
        if "code_interpreter" in features and features["code_interpreter"]:
            form_data["messages"] = add_or_update_user_message(
                (
                    request.app.state.config.CODE_INTERPRETER_PROMPT_TEMPLATE
                    if request.app.state.config.CODE_INTERPRETER_PROMPT_TEMPLATE != ""
                    else DEFAULT_CODE_INTERPRETER_PROMPT
                ),
                form_data["messages"],
            )
    # 准备工具 ID 和文件列表，用于后续的工具调用和 RAG 处理。
    tool_ids = form_data.pop("tool_ids", None)
    files = form_data.pop("files", None)
    # 去重是为了避免重复处理相同的文件/知识库。
    # Remove files duplicates
    if files:
        files = list({json.dumps(f, sort_keys=True): f for f in files}.values())
    # 确保处理后的工具 ID 和文件列表在 metadata 中可用，并将其传递给后续步骤（如 generate_chat_completion）。
    metadata = {
        **metadata,
        "tool_ids": tool_ids,
        "files": files,
    }
    form_data["metadata"] = metadata

    # Server side tools
    tool_ids = metadata.get("tool_ids", None)
    # Client side tools
    tool_servers = metadata.get("tool_servers", None)

    log.debug(f"{tool_ids=}")
    log.debug(f"{tool_servers=}")
    # 这个字典将用于存储所有准备好的工具信息，包括内部工具和外部工具服务器的函数定义、可调用包装器 (callable) 和规格 (spec)。
    # 键是工具函数的名称。
    tools_dict = {}

    if tool_ids:
        # get_tools 负责加载指定的内部 Python 工具模块。
        # 对于每个工具函数，它会创建一个异步包装器 (callable)，该包装器在调用时会注入 extra_params（如用户信息、请求对象等）。
        # 它还会获取或生成工具函数的规格说明 (spec)。
        # 返回的 tools_dict 包含了这些工具函数的信息，键是函数名。
        tools_dict = get_tools(
            request,
            tool_ids,
            user,
            {
                **extra_params,
                "__model__": models[task_model_id],
                "__messages__": form_data["messages"],
                "__files__": metadata.get("files", []),
            },
        )

    if tool_servers:
        # 对于每个服务器，提取其提供的工具函数规格 (specs)。
        # 将每个外部工具函数的信息添加到 tools_dict 中。
        # direct: True 标志表明这是一个外部工具。
        # server 字段存储了服务器的 URL 和可能的认证信息。
        # 注意: 这种直接传递 tool_servers 的方式不如通过配置 (TOOL_SERVER_CONNECTIONS) 添加外部工具常见。
        # 通过配置添加的外部工具是在 get_tools 函数内部处理的（当 tool_id 是 server:<idx> 格式时），
        # get_tools 会为它们创建 callable 包装器来调用 execute_tool_server。
        for tool_server in tool_servers:
            tool_specs = tool_server.pop("specs", [])

            for tool in tool_specs:
                tools_dict[tool["name"]] = {
                    "spec": tool,
                    "direct": True,
                    "server": tool_server,
                }

    if tools_dict:
        # 原生工具调用模式: 如果是 "native"，表示希望 LLM 直接处理工具调用决策。
        # 将完整的 tools_dict (包含 callable 等) 存储在 metadata["tools"] 中。
        # 这将在 process_chat_response 中用于实际执行 LLM 返回的工具调用指令。
        # 将工具的规格 (spec) 提取出来，格式化成 LLM（如 OpenAI）能理解的 tools 参数格式，
        # 并添加到 form_data["tools"] 中。这将随聊天请求一起发送给 LLM。

        if metadata.get("function_calling") == "native":
            # If the function calling is native, then call the tools function calling handler
            metadata["tools"] = tools_dict
            form_data["tools"] = [
                {"type": "function", "function": tool.get("spec", {})}
                for tool in tools_dict.values()
            ]
        else:
            # 非原生工具调用模式
            # 将工具执行结果添加到消息历史 (form_data["messages"]) 或 RAG 上下文 (sources) 中
            # 关键区别: 这种模式下，工具调用发生在主 LLM 调用之前，并且工具调用的决策是由一个单独的（可能是专门的）LLM 调用完成的。
            try:
                form_data, flags = await chat_completion_tools_handler(
                    request, form_data, extra_params, user, models, tools_dict
                )
                # 将 chat_completion_tools_handler 返回的 sources 添加到当前的 sources 列表中。
                sources.extend(flags.get("sources", []))

            except Exception as e:
                log.exception(e)
        log.debug(f"function_calling = {metadata.get('function_calling')}, form_data = {form_data}")
    try:
        # 处理 RAG（Retrieval-Augmented Generation）的核心步骤。
        form_data, flags = await chat_completion_files_handler(request, form_data, user)
        sources.extend(flags.get("sources", []))
    except Exception as e:
        log.exception(e)

    # 如果 sources 列表不为空（表示 RAG 或工具调用产生了上下文信息）。
    if len(sources) > 0:
        context_string = ""
        citated_file_idx = {}
        # 构建 context_string：遍历 sources，将每个文档片段 (doc_context) 用 <source id="..."> 标签包裹起来。id 用于后续可能的引用标记。
        for _, source in enumerate(sources, 1):
            if "document" in source:
                for doc_context, doc_meta in zip(
                    source["document"], source["metadata"]
                ):
                    file_id = doc_meta.get("file_id")
                    if file_id not in citated_file_idx:
                        citated_file_idx[file_id] = len(citated_file_idx) + 1
                    context_string += f'<source id="{citated_file_idx[file_id]}">{doc_context}</source>\n'

        context_string = context_string.strip()
        prompt = get_last_user_message(form_data["messages"])

        if prompt is None:
            raise Exception("No user message found")
        if (
            request.app.state.config.RELEVANCE_THRESHOLD == 0
            and context_string.strip() == ""
        ):
            log.debug(
                f"With a 0 relevancy threshold for RAG, the context cannot be empty"
            )

        # 使用配置的 RAG 模板 (RAG_TEMPLATE)、构建的 context_string 和用户 prompt 来生成最终注入到 LLM 请求中的上下文提示。
        # Workaround for Ollama 2.0+ system prompt issue
        # TODO: replace with add_or_update_system_message
        if model.get("owned_by") == "ollama":
            # 对于 Ollama 模型，使用 prepend_to_first_user_message_content (在 utils/misc.py) 将其添加到第一条用户消息的开头（作为对 Ollama 处理系统消息问题的变通）
            form_data["messages"] = prepend_to_first_user_message_content(
                rag_template(
                    request.app.state.config.RAG_TEMPLATE, context_string, prompt
                ),
                form_data["messages"],
            )
        else:
            # 对于其他模型，使用 add_or_update_system_message (在 utils/misc.py) 将其作为系统消息添加或更新。
            form_data["messages"] = add_or_update_system_message(
                rag_template(
                    request.app.state.config.RAG_TEMPLATE, context_string, prompt
                ),
                form_data["messages"],
            )

    # If there are citations, add them to the data_items
    # 过滤 sources 列表，确保每个来源都有一个名称。如果过滤后仍有来源，则将整个 sources 列表作为一个事件添加到 events 列表中。
    # 作用: 准备将检索到的来源信息发送给前端，以便在 UI 中显示引用。
    log.debug(f"Sources: {sources}")
    sources = [source for source in sources if source.get("source", {}).get("name", "")]
    log.debug(f"Sources (filter): {sources}")
    if len(sources) > 0:
        events.append({"sources": sources})
    # 如果模型关联了知识库 (model_knowledge 为真)。
    # 通过 event_emitter 向前端发送一个最终的状态更新事件，标记知识库搜索完成（即使没有找到来源，也标记为完成）。hidden: True 可能表示这个状态更新不在 UI 中显式显示给用户。
    if model_knowledge:
        await event_emitter(
            {
                "type": "status",
                "data": {
                    "action": "knowledge_search",
                    "query": user_message,
                    "done": True,
                    "hidden": True,
                },
            }
        )

    return form_data, metadata, events


async def process_chat_response(
    request, response, form_data, user, metadata, model, events, tasks
):
    async def background_tasks_handler():
        message_map = Chats.get_messages_by_chat_id(metadata["chat_id"])
        message = message_map.get(metadata["message_id"]) if message_map else None

        if message:
            messages = get_message_list(message_map, message.get("id"))

            if tasks and messages:
                if TASKS.TITLE_GENERATION in tasks:
                    if tasks[TASKS.TITLE_GENERATION]:
                        res = await generate_title(
                            request,
                            {
                                "model": message["model"],
                                "messages": messages,
                                "chat_id": metadata["chat_id"],
                            },
                            user,
                        )

                        if res and isinstance(res, dict):
                            if len(res.get("choices", [])) == 1:
                                title_string = (
                                    res.get("choices", [])[0]
                                    .get("message", {})
                                    .get("content", message.get("content", "New Chat"))
                                )
                            else:
                                title_string = ""

                            title_string = title_string[
                                title_string.find("{") : title_string.rfind("}") + 1
                            ]

                            try:
                                title = json.loads(title_string).get(
                                    "title", "New Chat"
                                )
                            except Exception as e:
                                title = ""

                            if not title:
                                title = messages[0].get("content", "New Chat")

                            Chats.update_chat_title_by_id(metadata["chat_id"], title)

                            await event_emitter(
                                {
                                    "type": "chat:title",
                                    "data": title,
                                }
                            )
                    elif len(messages) == 2:
                        title = messages[0].get("content", "New Chat")

                        Chats.update_chat_title_by_id(metadata["chat_id"], title)

                        await event_emitter(
                            {
                                "type": "chat:title",
                                "data": message.get("content", "New Chat"),
                            }
                        )

                if TASKS.TAGS_GENERATION in tasks and tasks[TASKS.TAGS_GENERATION]:
                    res = await generate_chat_tags(
                        request,
                        {
                            "model": message["model"],
                            "messages": messages,
                            "chat_id": metadata["chat_id"],
                        },
                        user,
                    )

                    if res and isinstance(res, dict):
                        if len(res.get("choices", [])) == 1:
                            tags_string = (
                                res.get("choices", [])[0]
                                .get("message", {})
                                .get("content", "")
                            )
                        else:
                            tags_string = ""

                        tags_string = tags_string[
                            tags_string.find("{") : tags_string.rfind("}") + 1
                        ]

                        try:
                            tags = json.loads(tags_string).get("tags", [])
                            Chats.update_chat_tags_by_id(
                                metadata["chat_id"], tags, user
                            )

                            await event_emitter(
                                {
                                    "type": "chat:tags",
                                    "data": tags,
                                }
                            )
                        except Exception as e:
                            pass

    event_emitter = None
    event_caller = None
    if (
        "session_id" in metadata
        and metadata["session_id"]
        and "chat_id" in metadata
        and metadata["chat_id"]
        and "message_id" in metadata
        and metadata["message_id"]
    ):
        event_emitter = get_event_emitter(metadata)
        event_caller = get_event_call(metadata)

    # Non-streaming response
    if not isinstance(response, StreamingResponse):
        if event_emitter:
            if "error" in response:
                error = response["error"].get("detail", response["error"])
                Chats.upsert_message_to_chat_by_id_and_message_id(
                    metadata["chat_id"],
                    metadata["message_id"],
                    {
                        "error": {"content": error},
                    },
                )

            if "selected_model_id" in response:
                Chats.upsert_message_to_chat_by_id_and_message_id(
                    metadata["chat_id"],
                    metadata["message_id"],
                    {
                        "selectedModelId": response["selected_model_id"],
                    },
                )

            choices = response.get("choices", [])
            if choices and choices[0].get("message", {}).get("content"):
                content = response["choices"][0]["message"]["content"]

                if content:

                    await event_emitter(
                        {
                            "type": "chat:completion",
                            "data": response,
                        }
                    )

                    title = Chats.get_chat_title_by_id(metadata["chat_id"])

                    await event_emitter(
                        {
                            "type": "chat:completion",
                            "data": {
                                "done": True,
                                "content": content,
                                "title": title,
                            },
                        }
                    )

                    # Save message in the database
                    Chats.upsert_message_to_chat_by_id_and_message_id(
                        metadata["chat_id"],
                        metadata["message_id"],
                        {
                            "content": content,
                        },
                    )

                    # Send a webhook notification if the user is not active
                    if get_active_status_by_user_id(user.id) is None:
                        webhook_url = Users.get_user_webhook_url_by_id(user.id)
                        if webhook_url:
                            post_webhook(
                                request.app.state.WEBUI_NAME,
                                webhook_url,
                                f"{title} - {request.app.state.config.WEBUI_URL}/c/{metadata['chat_id']}\n\n{content}",
                                {
                                    "action": "chat",
                                    "message": content,
                                    "title": title,
                                    "url": f"{request.app.state.config.WEBUI_URL}/c/{metadata['chat_id']}",
                                },
                            )

                    await background_tasks_handler()

            return response
        else:
            return response

    # Non standard response
    if not any(
        content_type in response.headers["Content-Type"]
        for content_type in ["text/event-stream", "application/x-ndjson"]
    ):
        return response

    extra_params = {
        "__event_emitter__": event_emitter,
        "__event_call__": event_caller,
        "__user__": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
        },
        "__metadata__": metadata,
        "__request__": request,
        "__model__": model,
    }
    filter_functions = [
        Functions.get_function_by_id(filter_id)
        for filter_id in get_sorted_filter_ids(model)
    ]

    # 流式响应处理
    if event_emitter and event_caller:
        task_id = str(uuid4())  # Create a unique task ID.
        model_id = form_data.get("model", "")

        Chats.upsert_message_to_chat_by_id_and_message_id(
            metadata["chat_id"],
            metadata["message_id"],
            {
                "model": model_id,
            },
        )

        def split_content_and_whitespace(content):
            content_stripped = content.rstrip()
            original_whitespace = (
                content[len(content_stripped) :]
                if len(content) > len(content_stripped)
                else ""
            )
            return content_stripped, original_whitespace

        def is_opening_code_block(content):
            backtick_segments = content.split("```")
            # Even number of segments means the last backticks are opening a new block
            return len(backtick_segments) > 1 and len(backtick_segments) % 2 == 0

        # 异步处理数据流，并在流结束后执行工具调用和可能的递归 LLM 调用。
        async def post_response_handler(response, events):
            def serialize_content_blocks(content_blocks, raw=False):
                content = ""

                for block in content_blocks:
                    if block["type"] == "text":
                        content = f"{content}{block['content'].strip()}\n"
                    elif block["type"] == "tool_calls":
                        attributes = block.get("attributes", {})

                        tool_calls = block.get("content", [])
                        results = block.get("results", [])

                        if results:

                            tool_calls_display_content = ""
                            for tool_call in tool_calls:

                                tool_call_id = tool_call.get("id", "")
                                tool_name = tool_call.get("function", {}).get(
                                    "name", ""
                                )
                                tool_arguments = tool_call.get("function", {}).get(
                                    "arguments", ""
                                )

                                tool_result = None
                                tool_result_files = None
                                for result in results:
                                    if tool_call_id == result.get("tool_call_id", ""):
                                        tool_result = result.get("content", None)
                                        tool_result_files = result.get("files", None)
                                        break

                                if tool_result:
                                    tool_calls_display_content = f'{tool_calls_display_content}\n<details type="tool_calls" done="true" id="{tool_call_id}" name="{tool_name}" arguments="{html.escape(json.dumps(tool_arguments))}" result="{html.escape(json.dumps(tool_result))}" files="{html.escape(json.dumps(tool_result_files)) if tool_result_files else ""}">\n<summary>Tool Executed</summary>\n</details>\n'
                                else:
                                    tool_calls_display_content = f'{tool_calls_display_content}\n<details type="tool_calls" done="false" id="{tool_call_id}" name="{tool_name}" arguments="{html.escape(json.dumps(tool_arguments))}">\n<summary>Executing...</summary>\n</details>'

                            if not raw:
                                content = f"{content}\n{tool_calls_display_content}\n\n"
                        else:
                            tool_calls_display_content = ""

                            for tool_call in tool_calls:
                                tool_call_id = tool_call.get("id", "")
                                tool_name = tool_call.get("function", {}).get(
                                    "name", ""
                                )
                                tool_arguments = tool_call.get("function", {}).get(
                                    "arguments", ""
                                )

                                tool_calls_display_content = f'{tool_calls_display_content}\n<details type="tool_calls" done="false" id="{tool_call_id}" name="{tool_name}" arguments="{html.escape(json.dumps(tool_arguments))}">\n<summary>Executing...</summary>\n</details>'

                            if not raw:
                                content = f"{content}\n{tool_calls_display_content}\n\n"

                    elif block["type"] == "reasoning":
                        reasoning_display_content = "\n".join(
                            (f"> {line}" if not line.startswith(">") else line)
                            for line in block["content"].splitlines()
                        )

                        reasoning_duration = block.get("duration", None)

                        if reasoning_duration is not None:
                            if raw:
                                content = f'{content}\n<{block["start_tag"]}>{block["content"]}<{block["end_tag"]}>\n'
                            else:
                                content = f'{content}\n<details type="reasoning" done="true" duration="{reasoning_duration}">\n<summary>Thought for {reasoning_duration} seconds</summary>\n{reasoning_display_content}\n</details>\n'
                        else:
                            if raw:
                                content = f'{content}\n<{block["start_tag"]}>{block["content"]}<{block["end_tag"]}>\n'
                            else:
                                content = f'{content}\n<details type="reasoning" done="false">\n<summary>Thinking…</summary>\n{reasoning_display_content}\n</details>\n'

                    elif block["type"] == "code_interpreter":
                        attributes = block.get("attributes", {})
                        output = block.get("output", None)
                        lang = attributes.get("lang", "")

                        content_stripped, original_whitespace = (
                            split_content_and_whitespace(content)
                        )
                        if is_opening_code_block(content_stripped):
                            # Remove trailing backticks that would open a new block
                            content = (
                                content_stripped.rstrip("`").rstrip()
                                + original_whitespace
                            )
                        else:
                            # Keep content as is - either closing backticks or no backticks
                            content = content_stripped + original_whitespace

                        if output:
                            output = html.escape(json.dumps(output))

                            if raw:
                                content = f'{content}\n<code_interpreter type="code" lang="{lang}">\n{block["content"]}\n</code_interpreter>\n```output\n{output}\n```\n'
                            else:
                                content = f'{content}\n<details type="code_interpreter" done="true" output="{output}">\n<summary>Analyzed</summary>\n```{lang}\n{block["content"]}\n```\n</details>\n'
                        else:
                            if raw:
                                content = f'{content}\n<code_interpreter type="code" lang="{lang}">\n{block["content"]}\n</code_interpreter>\n'
                            else:
                                content = f'{content}\n<details type="code_interpreter" done="false">\n<summary>Analyzing...</summary>\n```{lang}\n{block["content"]}\n```\n</details>\n'

                    else:
                        block_content = str(block["content"]).strip()
                        content = f"{content}{block['type']}: {block_content}\n"

                return content.strip()

            def convert_content_blocks_to_messages(content_blocks):
                messages = []

                temp_blocks = []
                for idx, block in enumerate(content_blocks):
                    if block["type"] == "tool_calls":
                        messages.append(
                            {
                                "role": "assistant",
                                "content": serialize_content_blocks(temp_blocks),
                                "tool_calls": block.get("content"),
                            }
                        )

                        results = block.get("results", [])

                        for result in results:
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": result["tool_call_id"],
                                    "content": result["content"],
                                }
                            )
                        temp_blocks = []
                    else:
                        temp_blocks.append(block)

                if temp_blocks:
                    content = serialize_content_blocks(temp_blocks)
                    if content:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": content,
                            }
                        )

                return messages

            def tag_content_handler(content_type, tags, content, content_blocks):
                end_flag = False

                def extract_attributes(tag_content):
                    """Extract attributes from a tag if they exist."""
                    attributes = {}
                    if not tag_content:  # Ensure tag_content is not None
                        return attributes
                    # Match attributes in the format: key="value" (ignores single quotes for simplicity)
                    matches = re.findall(r'(\w+)\s*=\s*"([^"]+)"', tag_content)
                    for key, value in matches:
                        attributes[key] = value
                    return attributes

                if content_blocks[-1]["type"] == "text":
                    for start_tag, end_tag in tags:
                        # Match start tag e.g., <tag> or <tag attr="value">
                        start_tag_pattern = rf"<{re.escape(start_tag)}(\s.*?)?>"
                        match = re.search(start_tag_pattern, content)
                        if match:
                            attr_content = (
                                match.group(1) if match.group(1) else ""
                            )  # Ensure it's not None
                            attributes = extract_attributes(
                                attr_content
                            )  # Extract attributes safely

                            # Capture everything before and after the matched tag
                            before_tag = content[
                                : match.start()
                            ]  # Content before opening tag
                            after_tag = content[
                                match.end() :
                            ]  # Content after opening tag

                            # Remove the start tag and after from the currently handling text block
                            content_blocks[-1]["content"] = content_blocks[-1][
                                "content"
                            ].replace(match.group(0) + after_tag, "")

                            if before_tag:
                                content_blocks[-1]["content"] = before_tag

                            if not content_blocks[-1]["content"]:
                                content_blocks.pop()

                            # Append the new block
                            content_blocks.append(
                                {
                                    "type": content_type,
                                    "start_tag": start_tag,
                                    "end_tag": end_tag,
                                    "attributes": attributes,
                                    "content": "",
                                    "started_at": time.time(),
                                }
                            )

                            if after_tag:
                                content_blocks[-1]["content"] = after_tag

                            break
                elif content_blocks[-1]["type"] == content_type:
                    start_tag = content_blocks[-1]["start_tag"]
                    end_tag = content_blocks[-1]["end_tag"]
                    # Match end tag e.g., </tag>
                    end_tag_pattern = rf"<{re.escape(end_tag)}>"

                    # Check if the content has the end tag
                    if re.search(end_tag_pattern, content):
                        end_flag = True

                        block_content = content_blocks[-1]["content"]
                        # Strip start and end tags from the content
                        start_tag_pattern = rf"<{re.escape(start_tag)}(.*?)>"
                        block_content = re.sub(
                            start_tag_pattern, "", block_content
                        ).strip()

                        end_tag_regex = re.compile(end_tag_pattern, re.DOTALL)
                        split_content = end_tag_regex.split(block_content, maxsplit=1)

                        # Content inside the tag
                        block_content = (
                            split_content[0].strip() if split_content else ""
                        )

                        # Leftover content (everything after `</tag>`)
                        leftover_content = (
                            split_content[1].strip() if len(split_content) > 1 else ""
                        )

                        if block_content:
                            content_blocks[-1]["content"] = block_content
                            content_blocks[-1]["ended_at"] = time.time()
                            content_blocks[-1]["duration"] = int(
                                content_blocks[-1]["ended_at"]
                                - content_blocks[-1]["started_at"]
                            )

                            # Reset the content_blocks by appending a new text block
                            if content_type != "code_interpreter":
                                if leftover_content:

                                    content_blocks.append(
                                        {
                                            "type": "text",
                                            "content": leftover_content,
                                        }
                                    )
                                else:
                                    content_blocks.append(
                                        {
                                            "type": "text",
                                            "content": "",
                                        }
                                    )

                        else:
                            # Remove the block if content is empty
                            content_blocks.pop()

                            if leftover_content:
                                content_blocks.append(
                                    {
                                        "type": "text",
                                        "content": leftover_content,
                                    }
                                )
                            else:
                                content_blocks.append(
                                    {
                                        "type": "text",
                                        "content": "",
                                    }
                                )

                        # Clean processed content
                        content = re.sub(
                            rf"<{re.escape(start_tag)}(.*?)>(.|\n)*?<{re.escape(end_tag)}>",
                            "",
                            content,
                            flags=re.DOTALL,
                        )

                return content, content_blocks, end_flag

            message = Chats.get_message_by_id_and_message_id(
                metadata["chat_id"], metadata["message_id"]
            )
            # 用于在流处理过程中 累积 从 LLM 返回的 tool_calls 指令块。
            tool_calls = []

            last_assistant_message = None
            try:
                if form_data["messages"][-1]["role"] == "assistant":
                    last_assistant_message = get_last_assistant_message(
                        form_data["messages"]
                    )
            except Exception as e:
                pass
            #  用于累积纯文本内容（主要用于非实时保存和可能的旧逻辑）
            content = (
                message.get("content", "")
                if message
                else last_assistant_message if last_assistant_message else ""
            )
            # 这是一个列表，用于结构化地存储响应内容。
            # 每个元素是一个字典，包含 type (如 "text", "tool_calls", "code_interpreter", "reasoning") 和 content。
            # 这对于后续重新构建消息列表至关重要。初始时通常有一个空的 "text" 块。
            content_blocks = [
                {
                    "type": "text",
                    "content": content,
                }
            ]
            # We might want to disable this by default
            DETECT_REASONING = True
            DETECT_SOLUTION = True
            DETECT_CODE_INTERPRETER = metadata.get("features", {}).get(
                "code_interpreter", False
            )

            reasoning_tags = [
                ("think", "/think"),
                ("thinking", "/thinking"),
                ("reason", "/reason"),
                ("reasoning", "/reasoning"),
                ("thought", "/thought"),
                ("Thought", "/Thought"),
                ("|begin_of_thought|", "|end_of_thought|"),
            ]

            code_interpreter_tags = [("code_interpreter", "/code_interpreter")]

            solution_tags = [("|begin_of_solution|", "|end_of_solution|")]

            try:
                for event in events:
                    await event_emitter(
                        {
                            "type": "chat:completion",
                            "data": event,
                        }
                    )

                    # Save message in the database
                    Chats.upsert_message_to_chat_by_id_and_message_id(
                        metadata["chat_id"],
                        metadata["message_id"],
                        {
                            **event,
                        },
                    )
                
                async def stream_body_handler(response):
                    nonlocal content
                    nonlocal content_blocks
                    # 累积工具调用信息: 将 id, function.name, function.arguments 等信息 逐步累加 到 response_tool_calls 列表中对应的工具调用对象里
                    response_tool_calls = []
                    # 迭代处理流
                    async for line in response.body_iterator:
                        line = line.decode("utf-8") if isinstance(line, bytes) else line
                        data = line
                        # Skip empty lines
                        if not data.strip():
                            continue

                        # "data:" is the prefix for each event
                        if not data.startswith("data:"):
                            continue

                        # Remove the prefix
                        data = data[len("data:") :].strip()

                        try:
                            data = json.loads(data)
                            # 应用流式过滤器
                            data, _ = await process_filter_functions(
                                request=request,
                                filter_functions=filter_functions,
                                filter_type="stream",
                                form_data=data,
                                extra_params=extra_params,
                            )
                            if data:
                                if "event" in data:
                                    await event_emitter(data.get("event", {}))

                                if "selected_model_id" in data:
                                    model_id = data["selected_model_id"]
                                    Chats.upsert_message_to_chat_by_id_and_message_id(
                                        metadata["chat_id"],
                                        metadata["message_id"],
                                        {
                                            "selectedModelId": model_id,
                                        },
                                    )
                                else:
                                    choices = data.get("choices", [])
                                    if not choices:
                                        error = data.get("error", {})
                                        if error:
                                            await event_emitter(
                                                {
                                                    "type": "chat:completion",
                                                    "data": {
                                                        "error": error,
                                                    },
                                                }
                                            )
                                        usage = data.get("usage", {})
                                        if usage:
                                            await event_emitter(
                                                {
                                                    "type": "chat:completion",
                                                    "data": {
                                                        "usage": usage,
                                                    },
                                                }
                                            )
                                        continue
                                    # 检测工具调用
                                    delta = choices[0].get("delta", {})
                                    delta_tool_calls = delta.get("tool_calls", None)

                                    if delta_tool_calls:
                                        for delta_tool_call in delta_tool_calls:
                                            tool_call_index = delta_tool_call.get(
                                                "index"
                                            )

                                            if tool_call_index is not None:
                                                # Check if the tool call already exists
                                                current_response_tool_call = None
                                                for (
                                                    response_tool_call
                                                ) in response_tool_calls:
                                                    if (
                                                        response_tool_call.get("index")
                                                        == tool_call_index
                                                    ):
                                                        current_response_tool_call = (
                                                            response_tool_call
                                                        )
                                                        break

                                                if current_response_tool_call is None:
                                                    # Add the new tool call
                                                    response_tool_calls.append(
                                                        delta_tool_call
                                                    )
                                                else:
                                                    # Update the existing tool call
                                                    delta_name = delta_tool_call.get(
                                                        "function", {}
                                                    ).get("name")
                                                    delta_arguments = (
                                                        delta_tool_call.get(
                                                            "function", {}
                                                        ).get("arguments")
                                                    )

                                                    if delta_name:
                                                        current_response_tool_call[
                                                            "function"
                                                        ]["name"] += delta_name

                                                    if delta_arguments:
                                                        current_response_tool_call[
                                                            "function"
                                                        ][
                                                            "arguments"
                                                        ] += delta_arguments

                                    value = delta.get("content")

                                    reasoning_content = delta.get(
                                        "reasoning_content"
                                    ) or delta.get("reasoning")
                                    if reasoning_content:
                                        if (
                                            not content_blocks
                                            or content_blocks[-1]["type"] != "reasoning"
                                        ):
                                            reasoning_block = {
                                                "type": "reasoning",
                                                "start_tag": "think",
                                                "end_tag": "/think",
                                                "attributes": {
                                                    "type": "reasoning_content"
                                                },
                                                "content": "",
                                                "started_at": time.time(),
                                            }
                                            content_blocks.append(reasoning_block)
                                        else:
                                            reasoning_block = content_blocks[-1]

                                        reasoning_block["content"] += reasoning_content

                                        data = {
                                            "content": serialize_content_blocks(
                                                content_blocks
                                            )
                                        }

                                    if value:
                                        if (
                                            content_blocks
                                            and content_blocks[-1]["type"]
                                            == "reasoning"
                                            and content_blocks[-1]
                                            .get("attributes", {})
                                            .get("type")
                                            == "reasoning_content"
                                        ):
                                            reasoning_block = content_blocks[-1]
                                            reasoning_block["ended_at"] = time.time()
                                            reasoning_block["duration"] = int(
                                                reasoning_block["ended_at"]
                                                - reasoning_block["started_at"]
                                            )

                                            content_blocks.append(
                                                {
                                                    "type": "text",
                                                    "content": "",
                                                }
                                            )

                                        content = f"{content}{value}"
                                        if not content_blocks:
                                            content_blocks.append(
                                                {
                                                    "type": "text",
                                                    "content": "",
                                                }
                                            )

                                        content_blocks[-1]["content"] = (
                                            content_blocks[-1]["content"] + value
                                        )
                                        # 处理特殊标签
                                        # 如果检测到 <think> 或类似标签，会创建一个新的 type: "reasoning" 的块，并将后续内容添加到这个块中，直到遇到结束标签。
                                        if DETECT_REASONING:
                                            content, content_blocks, _ = (
                                                tag_content_handler(
                                                    "reasoning",
                                                    reasoning_tags,
                                                    content,
                                                    content_blocks,
                                                )
                                            )
                                        # 如果检测到 <code_interpreter> 标签，会创建一个新的 type: "code_interpreter" 的块，并将代码内容添加到这个块中，直到遇到结束标签。
                                        if DETECT_CODE_INTERPRETER:
                                            content, content_blocks, end = (
                                                tag_content_handler(
                                                    "code_interpreter",
                                                    code_interpreter_tags,
                                                    content,
                                                    content_blocks,
                                                )
                                            )

                                            if end:
                                                break

                                        if DETECT_SOLUTION:
                                            content, content_blocks, _ = (
                                                tag_content_handler(
                                                    "solution",
                                                    solution_tags,
                                                    content,
                                                    content_blocks,
                                                )
                                            )
                                        # 序列化后的 content_blocks 保存到数据库。
                                        if ENABLE_REALTIME_CHAT_SAVE:
                                            # Save message in the database
                                            Chats.upsert_message_to_chat_by_id_and_message_id(
                                                metadata["chat_id"],
                                                metadata["message_id"],
                                                {
                                                    "content": serialize_content_blocks(
                                                        content_blocks
                                                    ),
                                                },
                                            )
                                        else:
                                            data = {
                                                "content": serialize_content_blocks(
                                                    content_blocks
                                                ),
                                            }
                                # 调用 event_emitter，将基于 content_blocks 序列化后的内容 (serialize_content_blocks(content_blocks)) 发送给前端，实现打字机和结构化内容的实时显示。
                                await event_emitter(
                                    {
                                        "type": "chat:completion",
                                        "data": data,
                                    }
                                )
                        except Exception as e:
                            done = "data: [DONE]" in line
                            if done:
                                pass
                            else:
                                log.debug("Error: ", e)
                                continue

                    if content_blocks:
                        # Clean up the last text block
                        if content_blocks[-1]["type"] == "text":
                            content_blocks[-1]["content"] = content_blocks[-1][
                                "content"
                            ].strip()

                            if not content_blocks[-1]["content"]:
                                content_blocks.pop()

                                if not content_blocks:
                                    content_blocks.append(
                                        {
                                            "type": "text",
                                            "content": "",
                                        }
                                    )

                    if response_tool_calls:
                        tool_calls.append(response_tool_calls)

                    if response.background:
                        await response.background()

                await stream_body_handler(response)

                MAX_TOOL_CALL_RETRIES = 10
                tool_call_retries = 0

                while len(tool_calls) > 0 and tool_call_retries < MAX_TOOL_CALL_RETRIES:
                    tool_call_retries += 1
                    # 获取在流处理期间累积完成的一批工具调用指令。
                    response_tool_calls = tool_calls.pop(0)

                    content_blocks.append(
                        {
                            "type": "tool_calls",
                            "content": response_tool_calls,
                        }
                    )
                    # event_emitter 发送状态更新，告知前端正在执行工具
                    await event_emitter(
                        {
                            "type": "chat:completion",
                            "data": {
                                "content": serialize_content_blocks(content_blocks),
                            },
                        }
                    )
                    # 获取之前 get_tools 准备好的工具字典。这个字典包含了外部工具服务器的包装器 callable 和规格 spec。
                    tools = metadata.get("tools", {})
                    # 用于存储工具执行结果。
                    results = []
                    for tool_call in response_tool_calls:
                        tool_call_id = tool_call.get("id", "")
                        tool_name = tool_call.get("function", {}).get("name", "")
                        # 使用 ast.literal_eval 或 json.loads 解析 arguments。ast.literal_eval 更健壮，能处理一些非严格 JSON 的情况。
                        tool_function_params = {}
                        try:
                            # json.loads cannot be used because some models do not produce valid JSON
                            tool_function_params = ast.literal_eval(
                                tool_call.get("function", {}).get("arguments", "{}")
                            )
                        except Exception as e:
                            log.debug(e)
                            # Fallback to JSON parsing
                            try:
                                tool_function_params = json.loads(
                                    tool_call.get("function", {}).get("arguments", "{}")
                                )
                            except Exception as e:
                                log.debug(
                                    f"Error parsing tool call arguments: {tool_call.get('function', {}).get('arguments', '{}')}"
                                )

                        tool_result = None

                        if tool_name in tools:
                            tool = tools[tool_name]
                            spec = tool.get("spec", {})

                            try:
                                allowed_params = (
                                    spec.get("parameters", {})
                                    .get("properties", {})
                                    .keys()
                                )

                                tool_function_params = {
                                    k: v
                                    for k, v in tool_function_params.items()
                                    if k in allowed_params
                                }

                                if tool.get("direct", False):
                                    tool_result = await event_caller(
                                        {
                                            "type": "execute:tool",
                                            "data": {
                                                "id": str(uuid4()),
                                                "name": tool_name,
                                                "params": tool_function_params,
                                                "server": tool.get("server", {}),
                                                "session_id": metadata.get(
                                                    "session_id", None
                                                ),
                                            },
                                        }
                                    )

                                else:
                                    # callable 是 get_tools 中通过 make_tool_function 创建的异步函数
                                    tool_function = tool["callable"]
                                    tool_result = await tool_function(
                                        **tool_function_params
                                    )

                            except Exception as e:
                                tool_result = str(e)

                        tool_result_files = []
                        if isinstance(tool_result, list):
                            for item in tool_result:
                                # check if string
                                if isinstance(item, str) and item.startswith("data:"):
                                    tool_result_files.append(item)
                                    tool_result.remove(item)

                        if isinstance(tool_result, dict) or isinstance(
                            tool_result, list
                        ):
                            tool_result = json.dumps(tool_result, indent=2)

                        results.append(
                            {
                                "tool_call_id": tool_call_id,
                                "content": tool_result,
                                **(
                                    {"files": tool_result_files}
                                    if tool_result_files
                                    else {}
                                ),
                            }
                        )

                    content_blocks[-1]["results"] = results

                    content_blocks.append(
                        {
                            "type": "text",
                            "content": "",
                        }
                    )
                    # 告知前端工具执行完成，并显示结果
                    await event_emitter(
                        {
                            "type": "chat:completion",
                            "data": {
                                "content": serialize_content_blocks(content_blocks),
                            },
                        }
                    )

                    try:
                        # 将更新后的消息列表（原始消息 + 助手工具调用 + 工具结果）再次发送给 LLM。
                        res = await generate_chat_completion(
                            request,
                            {
                                "model": model_id,
                                "stream": True,
                                "tools": form_data["tools"],
                                "messages": [
                                    *form_data["messages"],
                                    *convert_content_blocks_to_messages(content_blocks),
                                ],
                            },
                            user,
                        )

                        if isinstance(res, StreamingResponse):
                            await stream_body_handler(res)
                        else:
                            break
                    except Exception as e:
                        log.debug(e)
                        break

                if DETECT_CODE_INTERPRETER:
                    MAX_RETRIES = 5
                    retries = 0

                    while (
                        content_blocks[-1]["type"] == "code_interpreter"
                        and retries < MAX_RETRIES
                    ):
                        await event_emitter(
                            {
                                "type": "chat:completion",
                                "data": {
                                    "content": serialize_content_blocks(content_blocks),
                                },
                            }
                        )

                        retries += 1
                        log.debug(f"Attempt count: {retries}")

                        output = ""
                        try:
                            if content_blocks[-1]["attributes"].get("type") == "code":
                                code = content_blocks[-1]["content"]

                                if (
                                    request.app.state.config.CODE_INTERPRETER_ENGINE
                                    == "pyodide"
                                ):
                                    output = await event_caller(
                                        {
                                            "type": "execute:python",
                                            "data": {
                                                "id": str(uuid4()),
                                                "code": code,
                                                "session_id": metadata.get(
                                                    "session_id", None
                                                ),
                                            },
                                        }
                                    )
                                elif (
                                    request.app.state.config.CODE_INTERPRETER_ENGINE
                                    == "jupyter"
                                ):
                                    output = await execute_code_jupyter(
                                        request.app.state.config.CODE_INTERPRETER_JUPYTER_URL,
                                        code,
                                        (
                                            request.app.state.config.CODE_INTERPRETER_JUPYTER_AUTH_TOKEN
                                            if request.app.state.config.CODE_INTERPRETER_JUPYTER_AUTH
                                            == "token"
                                            else None
                                        ),
                                        (
                                            request.app.state.config.CODE_INTERPRETER_JUPYTER_AUTH_PASSWORD
                                            if request.app.state.config.CODE_INTERPRETER_JUPYTER_AUTH
                                            == "password"
                                            else None
                                        ),
                                        request.app.state.config.CODE_INTERPRETER_JUPYTER_TIMEOUT,
                                    )
                                else:
                                    output = {
                                        "stdout": "Code interpreter engine not configured."
                                    }

                                log.debug(f"Code interpreter output: {output}")

                                if isinstance(output, dict):
                                    stdout = output.get("stdout", "")

                                    if isinstance(stdout, str):
                                        stdoutLines = stdout.split("\n")
                                        for idx, line in enumerate(stdoutLines):
                                            if "data:image/png;base64" in line:
                                                id = str(uuid4())

                                                # ensure the path exists
                                                os.makedirs(
                                                    os.path.join(CACHE_DIR, "images"),
                                                    exist_ok=True,
                                                )

                                                image_path = os.path.join(
                                                    CACHE_DIR,
                                                    f"images/{id}.png",
                                                )

                                                with open(image_path, "wb") as f:
                                                    f.write(
                                                        base64.b64decode(
                                                            line.split(",")[1]
                                                        )
                                                    )

                                                stdoutLines[idx] = (
                                                    f"![Output Image {idx}](/cache/images/{id}.png)"
                                                )

                                        output["stdout"] = "\n".join(stdoutLines)

                                    result = output.get("result", "")

                                    if isinstance(result, str):
                                        resultLines = result.split("\n")
                                        for idx, line in enumerate(resultLines):
                                            if "data:image/png;base64" in line:
                                                id = str(uuid4())

                                                # ensure the path exists
                                                os.makedirs(
                                                    os.path.join(CACHE_DIR, "images"),
                                                    exist_ok=True,
                                                )

                                                image_path = os.path.join(
                                                    CACHE_DIR,
                                                    f"images/{id}.png",
                                                )

                                                with open(image_path, "wb") as f:
                                                    f.write(
                                                        base64.b64decode(
                                                            line.split(",")[1]
                                                        )
                                                    )

                                                resultLines[idx] = (
                                                    f"![Output Image {idx}](/cache/images/{id}.png)"
                                                )

                                        output["result"] = "\n".join(resultLines)
                        except Exception as e:
                            output = str(e)

                        content_blocks[-1]["output"] = output

                        content_blocks.append(
                            {
                                "type": "text",
                                "content": "",
                            }
                        )

                        await event_emitter(
                            {
                                "type": "chat:completion",
                                "data": {
                                    "content": serialize_content_blocks(content_blocks),
                                },
                            }
                        )

                        try:
                            res = await generate_chat_completion(
                                request,
                                {
                                    "model": model_id,
                                    "stream": True,
                                    "messages": [
                                        *form_data["messages"],
                                        {
                                            "role": "assistant",
                                            "content": serialize_content_blocks(
                                                content_blocks, raw=True
                                            ),
                                        },
                                    ],
                                },
                                user,
                            )

                            if isinstance(res, StreamingResponse):
                                await stream_body_handler(res)
                            else:
                                break
                        except Exception as e:
                            log.debug(e)
                            break

                title = Chats.get_chat_title_by_id(metadata["chat_id"])
                data = {
                    "done": True,
                    "content": serialize_content_blocks(content_blocks),
                    "title": title,
                }

                if not ENABLE_REALTIME_CHAT_SAVE:
                    # Save message in the database
                    Chats.upsert_message_to_chat_by_id_and_message_id(
                        metadata["chat_id"],
                        metadata["message_id"],
                        {
                            "content": serialize_content_blocks(content_blocks),
                        },
                    )

                # Send a webhook notification if the user is not active
                if get_active_status_by_user_id(user.id) is None:
                    webhook_url = Users.get_user_webhook_url_by_id(user.id)
                    if webhook_url:
                        post_webhook(
                            request.app.state.WEBUI_NAME,
                            webhook_url,
                            f"{title} - {request.app.state.config.WEBUI_URL}/c/{metadata['chat_id']}\n\n{content}",
                            {
                                "action": "chat",
                                "message": content,
                                "title": title,
                                "url": f"{request.app.state.config.WEBUI_URL}/c/{metadata['chat_id']}",
                            },
                        )

                await event_emitter(
                    {
                        "type": "chat:completion",
                        "data": data,
                    }
                )

                await background_tasks_handler()
            except asyncio.CancelledError:
                log.warning("Task was cancelled!")
                await event_emitter({"type": "task-cancelled"})

                if not ENABLE_REALTIME_CHAT_SAVE:
                    # Save message in the database
                    Chats.upsert_message_to_chat_by_id_and_message_id(
                        metadata["chat_id"],
                        metadata["message_id"],
                        {
                            "content": serialize_content_blocks(content_blocks),
                        },
                    )

            if response.background is not None:
                await response.background()

        # 后台任务创建: post_response_handler 通过 create_task 在后台运行，处理整个流。
        task_id, _ = create_task(
            post_response_handler(response, events), id=metadata["chat_id"]
        )
        return {"status": True, "task_id": task_id}

    else:
        # Fallback to the original response
        async def stream_wrapper(original_generator, events):
            def wrap_item(item):
                return f"data: {item}\n\n"

            for event in events:
                event, _ = await process_filter_functions(
                    request=request,
                    filter_functions=filter_functions,
                    filter_type="stream",
                    form_data=event,
                    extra_params=extra_params,
                )

                if event:
                    yield wrap_item(json.dumps(event))

            async for data in original_generator:
                data, _ = await process_filter_functions(
                    request=request,
                    filter_functions=filter_functions,
                    filter_type="stream",
                    form_data=data,
                    extra_params=extra_params,
                )

                if data:
                    yield data

        return StreamingResponse(
            stream_wrapper(response.body_iterator, events),
            headers=dict(response.headers),
            background=response.background,
        )
