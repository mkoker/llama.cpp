#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory


def _run_test_function(func) -> tuple[bool, str]:
    sig = inspect.signature(func)
    kwargs = {}
    tmp_ctx = None
    if 'tmp_path' in sig.parameters:
        tmp_ctx = TemporaryDirectory()
        kwargs['tmp_path'] = Path(tmp_ctx.name)
    try:
        func(**kwargs)
        return True, ''
    except Exception:
        return False, traceback.format_exc()
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str]) -> int:
    files = [Path(a) for a in argv[1:] if a.endswith('.py')]
    if not files:
        print('no tests collected')
        return 5

    passed = 0
    failed = 0
    failures: list[str] = []

    for path in files:
        mod = _load_module(path)
        for name, obj in sorted(mod.__dict__.items()):
            if name.startswith('test_') and callable(obj):
                ok, detail = _run_test_function(obj)
                if ok:
                    passed += 1
                else:
                    failed += 1
                    failures.append(f'{path}:{name}\n{detail}')

    if failures:
        print('\n'.join(failures))
        print(f'{failed} failed, {passed} passed')
        return 1

    print(f'{passed} passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
