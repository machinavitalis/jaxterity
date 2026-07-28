# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Render a checkup run (plus its history) as a self-contained HTML report.

The template (``template.html``, shipped as package data) is a single file
with no external requests: styles and charts are inline, and the run data is
injected as a JSON literal. Everything visual is derived from the injected
:class:`~jaxterity.checkup.CheckupReport` dicts — the renderer adds nothing
the report does not contain.
"""

from __future__ import annotations

import importlib.resources
import json

from .report import CheckupReport

_PLACEHOLDER = "__CHECKUP_DATA_JSON__"


def render_report_html(
    report: CheckupReport, history: list[CheckupReport] | None = None
) -> str:
    """Return the report as a self-contained HTML page.

    Args
    ----
    report : CheckupReport
        The run to render.
    history : list[CheckupReport] | None
        Prior runs of the same unit, oldest first (typically
        :meth:`~jaxterity.checkup.CheckupStore.runs`). ``report`` is appended
        automatically if it is not the last entry.

    Returns
    -------
    str
        Complete HTML document text.
    """
    runs = list(history or [])
    if not runs or runs[-1].to_dict() != report.to_dict():
        runs.append(report)
    payload = {"runs": [r.to_dict() for r in runs]}
    # "</" must not appear verbatim inside an inline <script> block.
    data = json.dumps(payload).replace("</", "<\\/")
    template = (
        importlib.resources.files("jaxterity.checkup")
        .joinpath("template.html")
        .read_text()
    )
    return template.replace(_PLACEHOLDER, data)


__all__ = ["render_report_html"]
