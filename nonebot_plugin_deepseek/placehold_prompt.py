import re
import ast
from enum import Enum
from typing import Callable


class parse_flag(Enum):
    text = 0
    var = 1
    block = 2
    endblock = 3


def parse_template(template: str):
    pattern = re.compile(r"({%.*?%}|{{.*?}})", re.DOTALL)
    parts = pattern.split(template)
    tokens: list[tuple[parse_flag, str]] = []
    # print(parts)
    for part in parts:
        if not parts:
            continue
        elif part.startswith("{{"):
            content = part[2:-2].strip()
            tokens.append((parse_flag.var, content))

        elif part.startswith("{%"):
            content = part[2:-2].strip()
            if content.startswith("end"):
                tokens.append((parse_flag.endblock, content[3:]))
            else:
                tokens.append((parse_flag.block, content))
        else:
            if part:
                tokens.append((parse_flag.text, part))

    return tokens


def parse_attr_chain(expr: str):
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    parts = []
    node = tree.body
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        parts.reverse()
        return parts
    return None


def extract_variables(expr):
    """改进版变量提取函数，支持所有语法场景"""
    variables = set()

    # 尝试解析为属性链（如 session.user.name）
    parts = parse_attr_chain(expr)
    if parts:
        variables.add(parts[0])
        return variables

    # 通用AST解析
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        try:
            tree = ast.parse(expr, mode="exec")
        except SyntaxError:
            return variables

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            variables.add(node.id)
    return variables


def compile_template(template: str) -> Callable[..., str]:
    tokens = parse_template(template)
    # print(tokens)
    variables = set()
    iterables = set()

    # 收集所有需要上下文访问的变量
    for token_type, content in tokens:
        if token_type == parse_flag.var:
            variables.update(extract_variables(content))
        elif token_type == parse_flag.block:
            parts = content.split()
            if parts[0] == "for" and len(parts) >= 4 and parts[2] == "in":
                iter_expr = " ".join(parts[3:])
                iter_vars = extract_variables(iter_expr)
                variables.update(iter_vars)
                iterables.add(parts[3]) if len(parts) >= 4 else None
            elif parts[0] in ("if", "elif"):
                cond_vars = extract_variables(" ".join(parts[1:]))
                variables.update(cond_vars)
    # 生成代码
    code = [
        "def render(context):",
        '    def safe_get(obj, attr, default=""):',
        "        if isinstance(obj, dict):",
        "            return obj.get(attr, default)",
        "        else:",
        "            return getattr(obj, attr, default)",
        "    result = []",
        "    _ctx = context",
    ]

    indent_level = 1
    stack = []
    # print(tokens)
    # 变量初始化（全部通过safe_get访问）
    for var in variables:
        code.append(f'    _ctx["{var}"] = safe_get(context, "{var}")')

    indent_level = 1
    stack = []
    last_type: parse_flag = parse_flag.text
    for token_type, content in tokens:
        if token_type == parse_flag.text:
            # 转义特殊字符并保留换行控制
            if last_type in (parse_flag.block, parse_flag.endblock):
                content = content.replace("\n", "", 1)
            escaped = content.replace("\n", "\\n").replace('"', '\\"')
            line = "    " * indent_level + f'result.append("{escaped.rstrip()}")'
            code.append(line)
        elif token_type == parse_flag.var:
            parts = parse_attr_chain(content)
            if parts:
                # 生成链式访问代码
                access_chain = f'safe_get(_ctx, "{parts[0]}")'
                for part in parts[1:]:
                    access_chain = f'safe_get({access_chain}, "{part}")'
                code.append("    " * indent_level + f"result.append({access_chain})")
            else:
                code.append("    " * indent_level + f'result.append(safe_get(_ctx, "{content}"))')
        elif token_type == parse_flag.block:
            parts = content.split()
            if parts[0] == "for":
                # 生成迭代器
                loop_var = parts[1]
                iter_expr = " ".join(parts[3:])
                code.append("    " * indent_level + f'for {loop_var} in safe_get(_ctx, "{parts[3]}", []):')
                code.append("    " * indent_level + f'    _ctx["{loop_var}"] = {loop_var}')  # 注入循环变量到上下文
                stack.append("for")
                indent_level += 1
            elif parts[0] == "if":
                # 生成条件表达式
                cond = " ".join([f'safe_get(_ctx, "{word}")' if word.isidentifier() else word for word in parts[1:]])
                code.append("    " * indent_level + f"if {cond}:")
                stack.append("if")
                indent_level += 1
            elif parts[0] == "elif":
                cond = " ".join([f'safe_get(_ctx, "{word}")' if word.isidentifier() else word for word in parts[1:]])
                code.append("    " * (indent_level - 1) + f"elif {cond}:")
            elif parts[0] == "else":
                code.append("    " * (indent_level - 1) + "else:")
        elif token_type == parse_flag.endblock:
            if stack:
                stack.pop()
                indent_level = max(1, indent_level - 1)
        last_type = token_type

    code.append('    return "".join(result)')
    code_str = "\n".join(code)  # Python Code
    locals_dict = {}
    exec(code_str, globals(), locals_dict)
    return locals_dict["render"]
