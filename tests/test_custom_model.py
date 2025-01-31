import pytest
from pydantic import ValidationError


def test_custom_model():
    from nonebot_plugin_deepseek.config import CustomModel

    # 测试基础字段验证和默认值
    def test_default_values():
        model = CustomModel(name="deepseek-chat")
        assert model.max_tokens == 4090
        assert model.base_url == "https://api.deepseek.com"
        assert model.temperature == 1

    def test_reasoner_default_max_tokens():
        model = CustomModel(name="deepseek-reasoner")
        assert model.max_tokens == 4000

    def test_invalid_max_tokens_range():
        with pytest.raises(ValidationError):
            CustomModel(name="test", max_tokens=0)  # 必须 >1
        with pytest.raises(ValidationError):
            CustomModel(name="test", max_tokens=8192)  # 必须 <8192

    def test_field_ranges():
        with pytest.raises(ValidationError):
            CustomModel(name="test", frequency_penalty=3)  # 允许范围 [-2, 2]
        with pytest.raises(ValidationError):
            CustomModel(name="test", top_p=2)  # 允许范围 [0, 1]

    # 测试 stop 字段验证
    def test_valid_stop_values():
        # 字符串类型
        model = CustomModel(name="test", stop="stop_word")
        assert model.stop == "stop_word"

        # 列表类型（<=16个元素）
        model = CustomModel(name="test", stop=["stop1", "stop2"])
        assert model.stop == ["stop1", "stop2"]

    def test_stop_list_too_long():
        with pytest.raises(ValueError, match="最多允许设置 16 个字符"):
            CustomModel(name="test", stop=[f"word{i}" for i in range(17)])

    # 测试模型特定逻辑
    def test_deepseek_chat_temperature_warning(caplog):
        CustomModel(name="deepseek-chat", temperature=0.5, top_p=0.5)
        assert "不建议同时修改" in caplog.text

    def test_deepseek_reasoner_constraints():
        # 不支持 logprobs
        with pytest.raises(ValueError, match="不支持设置 logprobs"):
            CustomModel(name="deepseek-reasoner", logprobs=True)

        # 设置无效字段时抛出警告
        with pytest.warns(UserWarning) as record:
            CustomModel(name="deepseek-reasoner", temperature=0.5, presence_penalty=1)
        assert any("不支持设置" in str(warn.message) for warn in record.list)

    def test_top_logprobs_requires_logprobs():
        # 同时启用 logprobs 和 top_logprobs
        CustomModel(name="deepseek-chat", logprobs=True, top_logprobs=5)

        # 仅设置 top_logprobs 不设置 logprobs
        with pytest.raises(ValueError, match="logprobs 必须为 True"):
            CustomModel(name="deepseek-chat", top_logprobs=5)

        # 显式关闭 logprobs 但设置 top_logprobs
        with pytest.raises(ValueError, match="logprobs 必须为 True"):
            CustomModel(name="deepseek-chat", logprobs=False, top_logprobs=5)

    def test_logprobs_combinations(caplog):
        # 测试合法组合
        model = CustomModel(name="deepseek-chat", logprobs=True)
        assert model.logprobs is True
        assert model.top_logprobs is None

        # 测试带 top_logprobs 的合法组合
        model = CustomModel(name="deepseek-chat", logprobs=True, top_logprobs=10)
        assert model.top_logprobs == 10

        # 测试非法组合的异常消息
        with pytest.raises(ValueError, match="logprobs 必须为 True") as excinfo:
            CustomModel(name="deepseek-chat", top_logprobs=5)
        assert "logprobs 必须为 True" in str(excinfo.value)

        def test_reasoner_max_tokens_warning(caplog):
            CustomModel(name="deepseek-reasoner", max_tokens=8001)
            assert "最大为 8000" in caplog.text

    # 测试额外字段和配置
    def test_extra_fields_allowed():
        model = CustomModel(name="test", extra_field="value")  # type: ignore
        assert hasattr(model, "extra_field")

    # 测试验证器边界条件
    def test_temperature_top_p_combinations():
        # 合法组合
        CustomModel(name="test", temperature=0)  # 允许最小值
        CustomModel(name="test", top_p=0)  # 允许最小值
        CustomModel(name="test", temperature=2, top_p=1)  # 允许最大值

    def test_presence_penalty_boundary():
        # 边界值测试
        CustomModel(name="test", presence_penalty=-2)  # 最小值
        CustomModel(name="test", presence_penalty=2)  # 最大值
        with pytest.raises(ValidationError):
            CustomModel(name="test", presence_penalty=-3)
