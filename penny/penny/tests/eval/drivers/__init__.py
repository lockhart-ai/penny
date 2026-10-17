"""A new eval DRIVER (a ``*_eval`` pytest fixture) lives here, in its own module — never
appended to ``penny/tests/eval/conftest.py`` (#2395). ``penny/tests/conftest.py`` — the
SUITE'S ACTUAL ROOT, not the eval one — loads every module this package holds as a plugin
(see its ``pytest_plugins`` line), so a driver added here needs no edit to either conftest,
and two case PRs that each add a driver no longer land on the same lines.

It has to be the suite's root conftest that loads these: pytest only honours
``pytest_plugins`` in an "initial" conftest — one on the path pytest was invoked with — and
``penny/tests/eval/conftest.py`` is that only for a run scoped under ``penny/tests/eval/``,
never for a run of the whole suite (``penny/tests/``), which is what CI runs.

A driver module pulls whatever shared machinery it needs from ``penny.tests.eval.conftest``
by name, exactly as a case module already does — the driver itself is the only thing that
moves here; the helpers it calls (``_require_ask``, ``_declare_case``, ``_arms``, …) stay put.
"""
