#!/usr/bin/env python3
"""Preflight the compiled AAAI submission artifacts."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


class PreflightError(RuntimeError):
    pass


def run(*args: str) -> str:
    completed = subprocess.run(
        list(args),
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise PreflightError(f"required tool not found: {name}")


def pdf_info(path: Path) -> dict[str, str]:
    output = run("pdfinfo", str(path))
    info: dict[str, str] = {}
    for line in output.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    return info


def check_pdf(path: Path, *, checklist_required: bool) -> int:
    if not path.exists():
        raise PreflightError(f"missing PDF: {path}")
    info = pdf_info(path)
    pages = int(info["Pages"])
    page_size = info.get("Page size", "")
    if "612 x 792 pts" not in page_size:
        raise PreflightError(f"{path.name} is not US Letter: {page_size}")

    fonts = run("pdffonts", str(path))
    if re.search(r"Type\s*3", fonts, flags=re.IGNORECASE):
        raise PreflightError(f"Type 3 font found in {path.name}")

    text = run("pdftotext", str(path), "-")
    if checklist_required and "Reproducibility Checklist" not in text:
        raise PreflightError("main PDF does not contain the reproducibility checklist")
    return pages


def check_log(path: Path, *, local_checklist_required: bool = False) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "aaai2027.sty" not in text:
        raise PreflightError(f"official AAAI-27 style not loaded in {path.name}")
    if local_checklist_required and "(./ReproducibilityChecklist.tex" not in text:
        raise PreflightError("main build did not include paper/ReproducibilityChecklist.tex")
    forbidden = (
        "LaTeX Error",
        "Package aaai Error",
        "There were undefined references",
        "Citation `",
        "Reference `",
        "Overfull \\hbox",
        "Overfull \\vbox",
    )
    for marker in forbidden:
        if marker in text:
            raise PreflightError(f"{marker!r} found in {path.name}")


def check_source() -> None:
    source_paths = [
        PAPER / "main.tex",
        PAPER / "supplement.tex",
        *sorted((PAPER / "sections").glob("*.tex")),
        *sorted((PAPER / "supp_sections").glob("*.tex")),
    ]
    forbidden_patterns = {
        r"\\usepackage\{hyperref\}": "hyperref",
        r"\\usepackage\{geometry\}": "geometry",
        r"\\usepackage\{fullpage\}": "fullpage",
        r"\\newpage": "manual page break",
        r"\\clearpage": "manual page break",
        r"\\pagebreak": "manual page break",
        r"\\addtolength": "manual layout change",
        r"\\vspace\s*\{\s*-": "negative vertical spacing",
        r"\\vskip\s*-": "negative vertical spacing",
    }
    for source_path in source_paths:
        source = source_path.read_text(encoding="utf-8")
        for pattern, description in forbidden_patterns.items():
            if re.search(pattern, source):
                relative = source_path.relative_to(ROOT)
                raise PreflightError(f"forbidden {description} in {relative}")

    checklist = PAPER / "ReproducibilityChecklist.tex"
    if not checklist.exists() or "Type your response here" in "\n".join(
        checklist.read_text(encoding="utf-8").splitlines()[90:]
    ):
        raise PreflightError("reproducibility checklist is missing an answer")


def labeled_page(label: str) -> int:
    aux = (PAPER / "main.aux").read_text(encoding="utf-8", errors="replace")
    match = re.search(
        rf"\\newlabel\{{{re.escape(label)}\}}\{{\{{.*?\}}\{{(\d+)\}}", aux
    )
    if not match:
        raise PreflightError(f"could not locate page label: {label}")
    return int(match.group(1))


def main() -> int:
    for tool in ("pdfinfo", "pdffonts", "pdftotext"):
        require_tool(tool)
    check_source()
    check_log(PAPER / "main.log", local_checklist_required=True)
    check_log(PAPER / "supplement.log")
    main_pages = check_pdf(PAPER / "main.pdf", checklist_required=True)
    supplement_pages = check_pdf(PAPER / "supplement.pdf", checklist_required=False)
    technical_pages = labeled_page("lasttechnicalpage")
    core_pages = labeled_page("lastpagebeforechecklist")
    if technical_pages > 7:
        raise PreflightError(
            f"technical content occupies {technical_pages} pages; expected at most 7"
        )
    if core_pages > 9:
        raise PreflightError(
            f"paper through references occupies {core_pages} pages; expected at most 9"
        )

    required_figures = (
        "bandit_identification_width.pdf",
        "sequential_identification_width.pdf",
        "bandit_certificate.pdf",
        "sequential_certificate.pdf",
        "random_bandit_width_ratio.pdf",
        "sequential_efficiency.pdf",
        "sequential_crossfit_mse.pdf",
        "stochastic_tabular_crossfit_mse.pdf",
        "policy_library_selection.pdf",
    )
    missing = [
        name
        for name in required_figures
        if not (ROOT / "results" / "figures" / name).exists()
    ]
    if missing:
        raise PreflightError(f"missing figures: {', '.join(missing)}")

    print(
        "Preflight passed: "
        f"technical content={technical_pages} pages, "
        f"paper through references={core_pages} pages, "
        f"main with checklist={main_pages} pages, supplement={supplement_pages} pages, "
        "US Letter, official AAAI style, no Type 3 fonts, no unresolved references."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PreflightError, subprocess.CalledProcessError, KeyError, ValueError) as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
