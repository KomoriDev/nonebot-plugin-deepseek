import pytest
import nonebot
from nonebug import NONEBOT_INIT_KWARGS
from pytest_asyncio import is_async_test
from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter


def pytest_configure(config: pytest.Config):
    config.stash[NONEBOT_INIT_KWARGS] = {
        "driver": "~fastapi",
        "log_level": "DEBUG",
        "command_start": {"/", ""},
        "deepseek": {"api_key": "sk-xxx", "enable_models": [{"name": "deepseek-v4-flash"}]},
    }


def pytest_collection_modifyitems(items: list[pytest.Item]):
    pytest_asyncio_tests = (item for item in items if is_async_test(item))
    session_scope_marker = pytest.mark.asyncio(loop_scope="session")
    for async_test in pytest_asyncio_tests:
        async_test.add_marker(session_scope_marker, append=False)


@pytest.fixture(scope="session", autouse=True)
def _load_adapters(nonebug_init: None):
    driver = nonebot.get_driver()
    driver.register_adapter(OneBotV11Adapter)

    nonebot.load_from_toml("pyproject.toml")
