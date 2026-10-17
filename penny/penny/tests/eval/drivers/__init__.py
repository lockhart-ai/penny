"""A new eval DRIVER (a ``*_eval`` pytest fixture) lives here, in its own module — never
appended to ``conftest.py`` (#2395). ``conftest.py`` loads every module this package holds
as a plugin (see its ``pytest_plugins`` line), so a driver added here needs no edit to
``conftest.py``, and two case PRs that each add a driver no longer land on the same lines.

A driver module pulls whatever shared machinery it needs from ``penny.tests.eval.conftest``
by name, exactly as a case module already does — the driver itself is the only thing that
moves here; the helpers it calls (``_require_ask``, ``_declare_case``, ``_arms``, …) stay put.
"""
