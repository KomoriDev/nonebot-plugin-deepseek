import re
import sys
import json
import inspect
import functools
import importlib.util
from pathlib import Path
from collections.abc import Callable
from typing import Any, Union, Optional

from ..log import ds_logger
from ..schemas import ToolCalls


class FunctionRegistry:
    def __init__(self):
        self._registry = {}
        self._type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            Any: "any",
        }

    def load(self, *directories: str, base_dir: Optional[Union[Path, str]] = None) -> None:
        base_path = Path(base_dir).resolve() if base_dir else Path.cwd()

        if str(base_path) not in sys.path:
            sys.path.insert(0, str(base_path))

        for rel_dir in directories:
            dir_path = (base_path / rel_dir).resolve()

            if not (dir_path / "__init__.py").exists():
                ds_logger("WARNING", f"Skipping non-package directory: {dir_path}")
                continue

            for py_file in dir_path.glob("*.py"):
                if py_file.name == "__init__.py":
                    continue

                package_path = dir_path.relative_to(base_path)
                module_name = ".".join(package_path.parts + (py_file.stem,))

                if module_name in sys.modules:
                    continue

                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if not spec or not spec.loader:
                    continue

                try:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                except Exception as e:
                    ds_logger("ERROR", f"Failed to loaded {module_name}: {str(e)}")

    def register(self, name: Optional[str] = None, description: Optional[str] = None):
        def decorator(func: Callable):
            nonlocal name, description
            func_name = name or func.__name__

            sig = inspect.signature(func)
            parameters = self._parse_parameters(func, sig)

            func_description = description or self._parse_description(func.__doc__)  # type: ignore

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            self._registry[func_name] = {
                "name": func_name,
                "description": func_description,
                "raw_parameters": parameters,
                "func": wrapper,
            }
            ds_logger("DEBUG", f'Succeeded to load function "{func_name}"')
            return wrapper

        return decorator

    def _parse_description(self, docstring: str) -> str:
        return docstring.split("\n\n")[0].strip() if docstring else ""

    def _parse_parameters(self, func: Callable, sig: inspect.Signature) -> dict:
        param_docs = self._parse_param_docs(func.__doc__ or "")
        parameters = {}

        for name, param in sig.parameters.items():
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
            parameters[name] = {
                "type": param_type,
                "description": param_docs.get(name, ""),
                "required": param.default == inspect.Parameter.empty,
            }
        return parameters

    def _parse_param_docs(self, docstring: str) -> dict[str, str]:
        param_docs = {}
        if not docstring:
            return param_docs

        args_section = False
        for line in docstring.split("\n"):
            line = line.strip()
            if line.lower() in ("args:", "参数:"):
                args_section = True
                continue
            if args_section:
                if not line:
                    continue
                if match := re.match(r"^(\w+)(\s*\(.*\))?:\s*(.*)", line):
                    param_docs[match.group(1)] = match.group(3).strip()
                else:
                    args_section = False
        return param_docs

    def _convert_type(self, py_type: type) -> str:
        return self._type_mapping.get(py_type, "any")

    def to_json(self) -> list:
        result = []
        for func_name, info in self._registry.items():
            params = info["raw_parameters"]
            properties = {}
            required = []

            for name, param in params.items():
                properties[name] = {
                    "type": self._convert_type(param["type"]),
                    "description": param["description"],
                }
                if param["required"]:
                    required.append(name)

            func_schema = {
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
            result.append(func_schema)
        return result

    async def execute_tool_call(self, tool_call: ToolCalls) -> Any:
        func_name = tool_call.function.name
        func_info = self._registry.get(func_name)
        if not func_info:
            raise ValueError(f"Function '{func_name}' is not registered.")

        args: dict[str, Any] = json.loads(tool_call.function.arguments)
        converted_args = {}

        for param_name, param_spec in func_info["raw_parameters"].items():
            if param_name not in args:
                if param_spec["required"]:
                    raise ValueError(f"Missing required parameter: {param_name}")
                continue
            value = args[param_name]
            param_type = param_spec["type"]
            try:
                converted_value = self._convert_value(value, param_type)
            except Exception as e:
                raise ValueError(
                    f"Parameter '{param_name}' value {value} cannot be converted to {param_type}: {e}"  # noqa: E501
                ) from e
            converted_args[param_name] = converted_value

        func = func_info["func"]

        result = func(**converted_args)
        if inspect.isawaitable(result):
            result = await result

        ds_logger("DEBUG", f"Calling {func_name} function")
        return result

    def _convert_value(self, value: Any, param_type: type) -> Any:
        if param_type is Any:
            return value

        if param_type is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lower_val = value.lower()
                if lower_val == "true":
                    return True
                elif lower_val == "false":
                    return False
                else:
                    raise ValueError(f"Invalid boolean string: {value}")
            return bool(value)

        try:
            return param_type(value)
        except (TypeError, ValueError):
            if param_type in (int, float):
                return param_type(str(value))
            raise


registry = FunctionRegistry()
