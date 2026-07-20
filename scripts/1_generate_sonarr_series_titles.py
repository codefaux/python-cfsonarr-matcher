#!/usr/bin/env python3
import os
from pathlib import Path

from pyarr import SonarrAPI

output_file = Path(Path.cwd() / "tests/fixtures/test_sonarr_series_titles.repr")

sonarr = SonarrAPI(os.getenv("SONARR_URL"), os.getenv("SONARR_API"))

show_titles: list[tuple[str, int]] = []

for _s in sonarr.get_series():
    _t: tuple[str, int] = _s["title"], _s["id"]
    show_titles.append(_t)

print(show_titles)

with output_file.open("w", encoding="utf-8") as f:
    f.write(repr(show_titles))
    f.write("\n")

print(f"-- {output_file.relative_to(Path.cwd())} updated.")
