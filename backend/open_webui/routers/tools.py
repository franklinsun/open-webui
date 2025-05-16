import logging
from pathlib import Path
from typing import Optional
import time

from open_webui.models.tools import (
    ToolForm,
    ToolModel,
    ToolResponse,
    ToolUserResponse,
    Tools,
)
from open_webui.utils.plugin import load_tool_module_by_id, replace_imports
from open_webui.config import CACHE_DIR
from open_webui.constants import ERROR_MESSAGES
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from open_webui.models.users import UserModel # Import UserModel from models.users
from open_webui.utils.tools import get_tool_specs 
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.access_control import has_access, has_permission
from open_webui.env import SRC_LOG_LEVELS

from open_webui.utils.tools import get_tool_servers_data

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])


router = APIRouter()

############################
# GetTools
############################


@router.get("/", response_model=list[ToolUserResponse])
async def get_tools(request: Request, user=Depends(get_verified_user)):
    if not request.app.state.TOOL_SERVERS:
        # If the tool servers are not set, we need to set them
        # This is done only once when the server starts
        # This is done to avoid loading the tool servers every time

        request.app.state.TOOL_SERVERS = await get_tool_servers_data(
            request.app.state.config.TOOL_SERVER_CONNECTIONS
        )

    tools = Tools.get_tools()
    for server in request.app.state.TOOL_SERVERS:
        tools.append(
            ToolUserResponse(
                **{
                    "id": f"server:{server['idx']}",
                    "user_id": f"server:{server['idx']}",
                    "name": server["openapi"]
                    .get("info", {})
                    .get("title", "Tool Server"),
                    "meta": {
                        "description": server["openapi"]
                        .get("info", {})
                        .get("description", ""),
                    },
                    "access_control": request.app.state.config.TOOL_SERVER_CONNECTIONS[
                        server["idx"]
                    ]
                    .get("config", {})
                    .get("access_control", None),
                    "updated_at": int(time.time()),
                    "created_at": int(time.time()),
                }
            )
        )

    # Add user-defined external tool servers
    if user.settings and user.settings.ui and user.settings.ui["toolServers"]:
        user_tool_server_connections = user.settings.ui['toolServers']
        try:
            processed_servers_list = await get_tool_servers_data(user_tool_server_connections)
            if processed_servers_list:
                for server in processed_servers_list:
                    tools.append(
                        ToolUserResponse(
                            id=f"user-server:{user.id}:{server["idx"]}",
                            user_id=user.id, 
                            name=server["openapi"].get("info", {}).get("title", f"User Tool Server {server["idx"]+1}"),
                            meta={
                                "description": server["openapi"].get("info", {}).get("description", ""),
                                "url": server.get('url'),
                                "path": server.get('path'),
                            },
                            access_control=None, 
                            updated_at=int(time.time()),
                            created_at=int(time.time()),
                        )
                    )
        except Exception as e:
            log.error(f"Error processing user tool server for user {user.id}: {e}")

    if user.role != "admin":
        tools = [
            tool
            for tool in tools
            if tool.user_id == user.id
            or has_access(user.id, "read", tool.access_control)
        ]

    return tools


############################
# Refresh Tool Servers
############################

# 定义响应模型
class RefreshResponse(BaseModel):
    status: bool
    message: str
    refreshed_servers_count: int

@router.post("/servers/refresh", response_model=RefreshResponse)
async def refresh_tool_servers(
    request: Request,
    user=Depends(get_admin_user) # 确保只有管理员可以触发
):
    """
    触发外部工具服务器 OpenAPI 规范的刷新。
    这将从配置的服务器获取最新的 API 定义。
    """
    log.info(f"User {user.id} triggered tool server refresh.")
    try:
        # 获取当前的工具服务器连接配置
        connections = request.app.state.config.TOOL_SERVER_CONNECTIONS

        # 获取当前会话 token (如果需要，用于 session 认证的服务器)
        session_token = None
        if hasattr(request.state, 'token') and request.state.token:
             session_token = request.state.token.credentials

        # 调用核心函数来获取并处理 OpenAPI 数据
        refreshed_servers = await get_tool_servers_data(
            connections,
            session_token=session_token # 传递 session_token
        )

        # 更新应用状态中的服务器信息
        request.app.state.TOOL_SERVERS = refreshed_servers

        admin_servers_count = len(refreshed_servers)
        log.info(f"Successfully refreshed admin-configured tool servers. Found {admin_servers_count} active servers.")

        # 清除用户自定义工具服务器的缓存
        user_cache_cleared_count = 0
        if hasattr(request.app.state, "USER_TOOL_SERVERS_CACHE"):
            user_cache_cleared_count = len(request.app.state.USER_TOOL_SERVERS_CACHE)
            request.app.state.USER_TOOL_SERVERS_CACHE.clear()
            log.info(f"Cleared USER_TOOL_SERVERS_CACHE for {user_cache_cleared_count} users. User-specific tool lists will be re-fetched on next access.")
        else:
            log.info("USER_TOOL_SERVERS_CACHE not found or already empty.")


        # 返回成功响应
        return RefreshResponse(
            status=True,
            message=f"Successfully refreshed admin-configured tool servers ({admin_servers_count} active). User tool server caches (for {user_cache_cleared_count} users) also cleared.",
            refreshed_servers_count=admin_servers_count
        )
    except Exception as e:
        log.exception(f"Failed to refresh tool servers: {e}")
        # 如果过程中出现任何错误，则抛出 HTTP 500 错误
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh tool servers: {str(e)}"
        )

############################################
# Get Tool Server Specs by ID (server:<idx>)
############################################

@router.get("/servers/{server_id}/specs", status_code=status.HTTP_200_OK)
async def get_tool_server_specs_by_id(
    server_id: str,
    request: Request,
    user: UserModel = Depends(get_verified_user) 
):
    """
    获取指定外部工具服务器 (通过 server:<idx> ID) 提供的所有 API 方法规格。
    这些规格是从服务器的 OpenAPI 定义中处理得到的。
    """
    log.info(f"User {user.id} attempting to fetch specs for tool server ID: {server_id}")
    
    # 1. 验证 server_id 格式
    if not server_id.startswith("server:") and not server_id.startswith("user-server:"):
        log.warning(f"Invalid server ID format received: {server_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid server ID format. Expected 'server:<index>' or 'user-server:<user_id>:<index>'.",
        )
    
    if server_id.startswith("server:"):
        # Logic for admin-configured tool servers
        try:
            server_idx = int(server_id.split(":")[1])
        except (IndexError, ValueError):
            log.warning(f"Could not parse index from server ID: {server_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid server ID format. Index could not be parsed.",
            )

        if not request.app.state.TOOL_SERVERS:
            log.info("TOOL_SERVERS is empty, attempting to populate.")
            try:
                connections = request.app.state.config.TOOL_SERVER_CONNECTIONS
                session_token = None
                if hasattr(request.state, 'token') and request.state.token and hasattr(request.state.token, 'credentials'):
                    session_token = request.state.token.credentials
                request.app.state.TOOL_SERVERS = await get_tool_servers_data(
                    connections, session_token=session_token
                )
                log.info(f"Populated TOOL_SERVERS with {len(request.app.state.TOOL_SERVERS)} entries.")
            except Exception as e:
                log.error(f"Failed to populate TOOL_SERVERS during request: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to load tool server configurations."
                )

        tool_server_data = None
        for server_in_state in request.app.state.TOOL_SERVERS:
            if server_in_state.get("idx") == server_idx:
                tool_server_data = server_in_state
                break

        if tool_server_data is None:
            log.warning(f"Tool server with index {server_idx} not found in loaded servers.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool server with index {server_idx} not found or is disabled.",
            )

        specs = tool_server_data.get("specs", [])
        log.info(f"Found {len(specs)} specs for admin tool server ID: {server_id}")
        return JSONResponse(content=specs)

    elif server_id.startswith("user-server:"):
        # Logic for user-configured tool servers
        log.info(f"Fetching specs for user-defined tool server ID: {server_id}")
        try:
            parts = server_id.split(":")
            if len(parts) != 3:
                raise ValueError("Invalid user-server ID format. Expected 'user-server:<user_id>:<index>'")
            
            # The user_id in the tool_id string should match the authenticated user's ID
            tool_owner_id = parts[1]
            original_idx = int(parts[2])

            if user.id != tool_owner_id:
                log.warning(f"User {user.id} attempting to access tool server specs of user {tool_owner_id}. Forbidden.")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You are not authorized to access specs for this tool server."
                )

            if not (user.settings and user.settings.ui and user.settings.ui["toolServers"]):
                log.warning(f"User {user.id} has no tool server configurations.")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User has no configured tool servers."
                )

            user_tool_server_connections = user.settings.ui["toolServers"]
            if not (0 <= original_idx < len(user_tool_server_connections)):
                log.warning(f"Index {original_idx} out of bounds for user {user.id}'s tool servers (count: {len(user_tool_connections)}).")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Tool server definition at index {original_idx} not found for this user."
                )
            
            conn_data = user_tool_server_connections[original_idx]
            conn_data_dict = conn_data.model_dump(exclude_none=True) if hasattr(conn_data, 'model_dump') else dict(conn_data)

            user_cached_servers = request.app.state.USER_TOOL_SERVERS_CACHE.get(user.id, [])
            server_data = None

            if original_idx < len(user_cached_servers) and user_cached_servers[original_idx] is not None:
                cached_server = user_cached_servers[original_idx]
                if cached_server.get("original_config") == conn_data_dict:
                    server_data = cached_server
                    log.debug(f"Using cached specs for user tool server ID: {server_id}")

            if server_data is None:
                log.debug(f"Fetching specs for user tool server ID: {server_id}")
                single_connection_config_list = [{
                    "url": conn_data_dict.get('url', ''),
                    "path": conn_data_dict.get('path', 'openapi.json'),
                    "auth_type": conn_data_dict.get('auth_type'),
                    "key": conn_data_dict.get('key'),
                    "config": {**conn_data_dict.get('config', {}), "enable": True}
                }]
                effective_session_token = None
                if conn_data_dict.get('auth_type') == "session":
                    if hasattr(request.state, 'token') and request.state.token and hasattr(request.state.token, 'credentials'):
                        effective_session_token = request.state.token.credentials
                
                processed_servers_list = await get_tool_servers_data(
                    single_connection_config_list, session_token=effective_session_token
                )
                if processed_servers_list:
                    server_data = processed_servers_list[0]
                    server_data["original_config"] = conn_data_dict # Store for cache validation
                    # Update cache
                    if user.id not in request.app.state.USER_TOOL_SERVERS_CACHE:
                        request.app.state.USER_TOOL_SERVERS_CACHE[user.id] = []
                    user_cache_list = request.app.state.USER_TOOL_SERVERS_CACHE[user.id]
                    while len(user_cache_list) <= original_idx:
                        user_cache_list.append(None)
                    user_cache_list[original_idx] = server_data

            if server_data and "specs" in server_data:
                log.info(f"Found {len(server_data['specs'])} specs for user tool server ID: {server_id}")
                return JSONResponse(content=server_data["specs"])
            else:
                log.warning(f"No specs found or error fetching for user tool server ID: {server_id}")
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Could not retrieve specs for user tool server at index {original_idx}.")

        except ValueError as e:
            log.warning(f"Invalid user-server ID format or index: {server_id}. Error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid user-server ID format: {e}",
            )
        except Exception as e:
            log.error(f"Error fetching specs for user tool server {server_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error retrieving specs for user tool server: {str(e)}",
            )
    else:
        # Fallback for unrecognized ID format
        log.warning(f"Unrecognized server ID format: {server_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid server ID format.",
        )

############################
# GetToolList
############################


@router.get("/list", response_model=list[ToolUserResponse])
async def get_tool_list(user=Depends(get_verified_user)):
    if user.role == "admin":
        tools = Tools.get_tools()
    else:
        tools = Tools.get_tools_by_user_id(user.id, "write")
    return tools


############################
# ExportTools
############################


@router.get("/export", response_model=list[ToolModel])
async def export_tools(user=Depends(get_admin_user)):
    tools = Tools.get_tools()
    return tools


############################
# CreateNewTools
############################


@router.post("/create", response_model=Optional[ToolResponse])
async def create_new_tools(
    request: Request,
    form_data: ToolForm,
    user=Depends(get_verified_user),
):
    if user.role != "admin" and not has_permission(
        user.id, "workspace.tools", request.app.state.config.USER_PERMISSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    if not form_data.id.isidentifier():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only alphanumeric characters and underscores are allowed in the id",
        )

    form_data.id = form_data.id.lower()

    tools = Tools.get_tool_by_id(form_data.id)
    if tools is None:
        try:
            form_data.content = replace_imports(form_data.content)
            tool_module, frontmatter = load_tool_module_by_id(
                form_data.id, content=form_data.content
            )
            form_data.meta.manifest = frontmatter

            TOOLS = request.app.state.TOOLS
            TOOLS[form_data.id] = tool_module

            specs = get_tool_specs(TOOLS[form_data.id])
            tools = Tools.insert_new_tool(user.id, form_data, specs)

            tool_cache_dir = CACHE_DIR / "tools" / form_data.id
            tool_cache_dir.mkdir(parents=True, exist_ok=True)

            if tools:
                return tools
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ERROR_MESSAGES.DEFAULT("Error creating tools"),
                )
        except Exception as e:
            log.exception(f"Failed to load the tool by id {form_data.id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT(str(e)),
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ID_TAKEN,
        )


############################
# GetToolsById
############################


@router.get("/id/{id}", response_model=Optional[ToolModel])
async def get_tools_by_id(id: str, user=Depends(get_verified_user)):
    tools = Tools.get_tool_by_id(id)

    if tools:
        if (
            user.role == "admin"
            or tools.user_id == user.id
            or has_access(user.id, "read", tools.access_control)
        ):
            return tools
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# UpdateToolsById
############################


@router.post("/id/{id}/update", response_model=Optional[ToolModel])
async def update_tools_by_id(
    request: Request,
    id: str,
    form_data: ToolForm,
    user=Depends(get_verified_user),
):
    tools = Tools.get_tool_by_id(id)
    if not tools:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    # Is the user the original creator, in a group with write access, or an admin
    if (
        tools.user_id != user.id
        and not has_access(user.id, "write", tools.access_control)
        and user.role != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    try:
        form_data.content = replace_imports(form_data.content)
        tool_module, frontmatter = load_tool_module_by_id(id, content=form_data.content)
        form_data.meta.manifest = frontmatter

        TOOLS = request.app.state.TOOLS
        TOOLS[id] = tool_module

        specs = get_tool_specs(TOOLS[id])

        updated = {
            **form_data.model_dump(exclude={"id"}),
            "specs": specs,
        }

        log.debug(updated)
        tools = Tools.update_tool_by_id(id, updated)

        if tools:
            return tools
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT("Error updating tools"),
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


############################
# DeleteToolsById
############################


@router.delete("/id/{id}/delete", response_model=bool)
async def delete_tools_by_id(
    request: Request, id: str, user=Depends(get_verified_user)
):
    tools = Tools.get_tool_by_id(id)
    if not tools:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        tools.user_id != user.id
        and not has_access(user.id, "write", tools.access_control)
        and user.role != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    result = Tools.delete_tool_by_id(id)
    if result:
        TOOLS = request.app.state.TOOLS
        if id in TOOLS:
            del TOOLS[id]

    return result


############################
# GetToolValves
############################


@router.get("/id/{id}/valves", response_model=Optional[dict])
async def get_tools_valves_by_id(id: str, user=Depends(get_verified_user)):
    tools = Tools.get_tool_by_id(id)
    if tools:
        try:
            valves = Tools.get_tool_valves_by_id(id)
            return valves
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT(str(e)),
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# GetToolValvesSpec
############################


@router.get("/id/{id}/valves/spec", response_model=Optional[dict])
async def get_tools_valves_spec_by_id(
    request: Request, id: str, user=Depends(get_verified_user)
):
    tools = Tools.get_tool_by_id(id)
    if tools:
        if id in request.app.state.TOOLS:
            tools_module = request.app.state.TOOLS[id]
        else:
            tools_module, _ = load_tool_module_by_id(id)
            request.app.state.TOOLS[id] = tools_module

        if hasattr(tools_module, "Valves"):
            Valves = tools_module.Valves
            return Valves.schema()
        return None
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# UpdateToolValves
############################


@router.post("/id/{id}/valves/update", response_model=Optional[dict])
async def update_tools_valves_by_id(
    request: Request, id: str, form_data: dict, user=Depends(get_verified_user)
):
    tools = Tools.get_tool_by_id(id)
    if not tools:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        tools.user_id != user.id
        and not has_access(user.id, "write", tools.access_control)
        and user.role != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    if id in request.app.state.TOOLS:
        tools_module = request.app.state.TOOLS[id]
    else:
        tools_module, _ = load_tool_module_by_id(id)
        request.app.state.TOOLS[id] = tools_module

    if not hasattr(tools_module, "Valves"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )
    Valves = tools_module.Valves

    try:
        form_data = {k: v for k, v in form_data.items() if v is not None}
        valves = Valves(**form_data)
        Tools.update_tool_valves_by_id(id, valves.model_dump())
        return valves.model_dump()
    except Exception as e:
        log.exception(f"Failed to update tool valves by id {id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


############################
# ToolUserValves
############################


@router.get("/id/{id}/valves/user", response_model=Optional[dict])
async def get_tools_user_valves_by_id(id: str, user=Depends(get_verified_user)):
    tools = Tools.get_tool_by_id(id)
    if tools:
        try:
            user_valves = Tools.get_user_valves_by_id_and_user_id(id, user.id)
            return user_valves
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT(str(e)),
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


@router.get("/id/{id}/valves/user/spec", response_model=Optional[dict])
async def get_tools_user_valves_spec_by_id(
    request: Request, id: str, user=Depends(get_verified_user)
):
    tools = Tools.get_tool_by_id(id)
    if tools:
        if id in request.app.state.TOOLS:
            tools_module = request.app.state.TOOLS[id]
        else:
            tools_module, _ = load_tool_module_by_id(id)
            request.app.state.TOOLS[id] = tools_module

        if hasattr(tools_module, "UserValves"):
            UserValves = tools_module.UserValves
            return UserValves.schema()
        return None
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


@router.post("/id/{id}/valves/user/update", response_model=Optional[dict])
async def update_tools_user_valves_by_id(
    request: Request, id: str, form_data: dict, user=Depends(get_verified_user)
):
    tools = Tools.get_tool_by_id(id)

    if tools:
        if id in request.app.state.TOOLS:
            tools_module = request.app.state.TOOLS[id]
        else:
            tools_module, _ = load_tool_module_by_id(id)
            request.app.state.TOOLS[id] = tools_module

        if hasattr(tools_module, "UserValves"):
            UserValves = tools_module.UserValves

            try:
                form_data = {k: v for k, v in form_data.items() if v is not None}
                user_valves = UserValves(**form_data)
                Tools.update_user_valves_by_id_and_user_id(
                    id, user.id, user_valves.model_dump()
                )
                return user_valves.model_dump()
            except Exception as e:
                log.exception(f"Failed to update user valves by id {id}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ERROR_MESSAGES.DEFAULT(str(e)),
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )
