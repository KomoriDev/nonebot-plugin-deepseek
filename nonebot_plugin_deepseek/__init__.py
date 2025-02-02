from dataclasses import asdict
from importlib.util import find_spec

import httpx
from nonebot import require
from nonebot.adapters import Event
from nonebot.params import Depends
from nonebot.matcher import Matcher
from nonebot.permission import User, SuperUser, Permission
from nonebot.plugin import PluginMetadata, inherit_supported_adapters

require("nonebot_plugin_waiter")
require("nonebot_plugin_alconna")
require("nonebot_plugin_localstore")
from arclet.alconna import config as alc_config
from nonebot_plugin_waiter import Waiter, prompt
from nonebot_plugin_alconna.uniseg import UniMessage
from nonebot_plugin_alconna.builtins.extensions.reply import ReplyMergeExtension
from nonebot_plugin_alconna import (
    Args,
    Field,
    Match,
    Query,
    Option,
    Alconna,
    MultiVar,
    Namespace,
    Subcommand,
    CommandMeta,
    on_alconna,
)

if find_spec("nonebot_plugin_htmlrender"):
    require("nonebot_plugin_htmlrender")
    from nonebot_plugin_htmlrender import md_to_pic

    is_to_pic = True
else:
    is_to_pic = False

from .apis import API
from . import hook as hook
from .function_call import registry
from .exception import RequestException
from .extension import CleanDocExtension
from .utils import extract_content_and_think
from .config import Config, config, model_config

__plugin_meta__ = PluginMetadata(
    name="DeepSeek",
    description="接入 DeepSeek 模型，提供智能对话与问答功能",
    usage="/deepseek -h",
    type="application",
    config=Config,
    homepage="https://github.com/KomoriDev/nonebot-plugin-deepseek",
    supported_adapters=inherit_supported_adapters("nonebot_plugin_alconna"),
    extra={
        "unique_name": "DeepSeek",
        "author": "Komorebi <mute231010@gmail.com>",
        "version": "0.1.5",
    },
)

if not config.md_to_pic:
    is_to_pic = False

ns = Namespace("deepseek", disable_builtin_options=set())
alc_config.namespaces["deepseek"] = ns

deepseek = on_alconna(
    Alconna(
        "deepseek",
        Args["content?#内容", MultiVar("str")],
        Option(
            "--use-model",
            Args[
                "model#模型名称",
                config.get_enable_models(),
                Field(completion=lambda: f"请输入模型名，预期为：{config.get_enable_models()} 其中之一"),
            ],
            help_text="指定模型",
        ),
        Option("--with-context", help_text="启用多轮对话"),
        Subcommand("--balance", help_text="查看余额"),
        Subcommand(
            "model",
            Option("-l|--list", help_text="支持的模型列表"),
            Option(
                "--set-default",
                Args[
                    "model#模型名称",
                    config.get_enable_models(),
                    Field(completion=lambda: f"请输入模型名，预期为：{config.get_enable_models()} 其中之一"),
                ],
                dest="set",
                help_text="设置默认模型",
            ),
            help_text="模型相关设置",
        ),
        namespace=alc_config.namespaces["deepseek"],
        meta=CommandMeta(
            description=__plugin_meta__.description,
            usage=__plugin_meta__.usage,
        ),
    ),
    aliases={"ds"},
    use_cmd_start=True,
    skip_for_unmatch=False,
    comp_config={"lite": True},
    extensions=[ReplyMergeExtension, CleanDocExtension],
)

deepseek.shortcut("多轮对话", {"command": "deepseek --with-context", "fuzzy": True, "prefix": True})
deepseek.shortcut("深度思考", {"command": "deepseek --use-model deepseek-reasoner", "fuzzy": True, "prefix": True})
deepseek.shortcut("余额", {"command": "deepseek --balance", "fuzzy": False, "prefix": True})
deepseek.shortcut("模型列表", {"command": "deepseek model --list", "fuzzy": False, "prefix": True})
deepseek.shortcut("设置默认模型", {"command": "deepseek model --set-default", "fuzzy": True, "prefix": True})


@deepseek.assign("balance")
async def _(is_superuser: bool = Depends(SuperUser())):
    if not is_superuser:
        return
    try:
        balances = await API.query_balance()

        await deepseek.finish(
            "".join(
                f"""
                货币：{balance.currency}
                总的可用余额: {balance.total_balance}
                未过期的赠金余额: {balance.granted_balance}
                充值余额: {balance.topped_up_balance}
                """
                for balance in balances.balance_infos
            )
        )
    except ValueError as e:
        await deepseek.finish(str(e))


@deepseek.assign("model.list")
async def _():
    model_list = "\n".join(
        f"- {model}（默认）" if model == model_config.default_model else f"- {model}"
        for model in config.get_enable_models()
    )
    message = (
        f"支持的模型列表: \n{model_list}\n"
        "输入 `/deepseek [内容] --use-model [模型名]` 单次选择模型\n"
        "输入 `/deepseek model --set-default [模型名]` 设置默认模型"
    )
    await deepseek.finish(message)


@deepseek.assign("model.set")
async def _(
    is_superuser: bool = Depends(SuperUser()),
    model: Query[str] = Query("model.set.model"),
):
    if not is_superuser:
        return
    model_config.default_model = model.result
    model_config.save()
    await deepseek.finish(f"已设置默认模型为：{model.result}")


@deepseek.handle()
async def _(
    event: Event,
    matcher: Matcher,
    content: Match[tuple[str, ...]],
    model_name: Query[str] = Query("use-model.model"),
    context_option: Query[bool] = Query("with-context.value"),
):
    if not content.available:
        resp = await prompt("你想对 DeepSeek 说什么呢？", timeout=60)
        if resp is None:
            await deepseek.finish("等待超时")
        text = resp.extract_plain_text()
        if text in ["结束", "取消", "done"]:
            await deepseek.finish("已结束对话")
        chat_content = text
    else:
        chat_content = " ".join(content.result)

    if not model_name.available:
        model_name.result = model_config.default_model

    message = [{"role": "user", "content": chat_content}]

    try:
        if not context_option.available:
            # 单次对话处理保持不变
            completion = await API.chat(message, model=model_name.result)
            result = completion.choices[0].message
            if result.tool_calls:
                message.append(asdict(result))
                fc_result = await registry.execute_tool_call(result.tool_calls[0])
                message.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.tool_calls[0].id,
                        "content": fc_result,
                    }
                )
                completion = await API.chat(message, model=model_name.result)
                result = completion.choices[0].message

            ds_content, ds_think = extract_content_and_think(result)

            if is_to_pic:
                output = (
                    f"<blockquote><p> {ds_think} </p></blockquote>" + ds_content
                    if ds_think and config.enable_send_thinking and ds_content
                    else ds_content
                )
                unimsg = UniMessage.image(raw=await md_to_pic(output))  # type: ignore
                if unimsg:
                    await unimsg.finish()
                await deepseek.finish(output)
            else:
                output = (
                    ds_think + f"\n----\n{ds_content}"
                    if ds_think and config.enable_send_thinking and ds_content
                    else ds_content
                )
                await deepseek.finish(output)

        # 多轮对话处理
        def handler(event: Event):
            text = event.get_plaintext().strip().lower()
            if text in ["结束", "取消", "done"]:
                return False
            if text in ["回滚","回滚对话", "rollback"]:  # 检测回滚指令
                return "rollback"
            return text

        permission = Permission(User.from_event(event, perm=matcher.permission))
        waiter = Waiter(waits=["message"], handler=handler, matcher=deepseek, permission=permission)
        waiter.future.set_result("")

        async for resp in waiter(default=False, timeout=config.context_timeout):
            if resp is False:
                await deepseek.finish("已结束对话")
            if resp == "rollback":  # 处理回滚指令
                if len(message) > 1:  # 确保至少有两条消息（用户和助手）
                    # 回滚一轮对话
                    message.pop()  # 移除助手回复
                    message.pop()  # 移除用户输入
                    await deepseek.send("已回滚一轮对话，请继续输入")
                    continue
                else:
                    await deepseek.send("无法回滚，当前对话记录为空")
                    continue

            prev_length = len(message)  # 记录当前消息长度用于回滚
            if resp and isinstance(resp, str):
                message.append({"role": "user", "content": resp})

            try:
                completion = await API.chat(message, model=model_name.result)
            except httpx.ReadTimeout:
                # 回滚到API调用前的消息状态
                if prev_length == 1:
                    # 如果是第一轮对话，恢复到初始状态
                    message = []
                else:
                    # 否则，恢复到超时前的状态
                    message = message[:prev_length]
                await deepseek.send("网络超时，请重新输入")
                continue  # 继续等待用户输入
            except RequestException as e:
                await deepseek.finish(str(e))

            result = completion.choices[0].message
            ds_content, ds_think = extract_content_and_think(result)

            # 发送回复时，根据 config.enable_send_thinking 处理 think
            output = (
                ds_think + f"\n----\n{ds_content}"
                if ds_think and config.enable_send_thinking and ds_content
                else ds_content
            )
            if output:
                await deepseek.send(output)

            # 将助手回复添加到消息记录时，移除 think 部分
            result.reasoning_content = None  # 清空 think 内容
            message.append(asdict(result))

            # 处理工具调用
            if result.tool_calls:
                try:
                    fc_result = await registry.execute_tool_call(result.tool_calls[0])
                except Exception as e:
                    # 回滚助手回复并提示错误
                    message.pop()  # 移除未成功的助手回复
                    await deepseek.send(f"工具调用失败：{str(e)}")
                    continue
                message.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.tool_calls[0].id,
                        "content": fc_result,
                    }
                )
                # 重置等待用户输入
                waiter.future.set_result("")
                continue

    except httpx.ReadTimeout:
        # 处理外层超时（单次对话情况）
        await deepseek.finish("网络超时，再试试吧")
    except RequestException as e:
        await deepseek.finish(str(e))
