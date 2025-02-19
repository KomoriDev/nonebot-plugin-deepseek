import sys


def main(*args):
    from ..cli import deepseek

    deepseek.main(*(["nb deepseek"] + list(args or sys.argv[1:])))
