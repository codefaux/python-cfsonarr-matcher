import ast
import hashlib
import marshal
import os
from pathlib import Path

from cfsonarrmatcher import match_to_show
from pyarr import SonarrAPI


def sonarr_series_titles() -> list[str]:
    sonarr = SonarrAPI(os.getenv("SONARR_URL"), os.getenv("SONARR_API"))

    show_titles = []
    for _s in sonarr.get_series():
        show_titles.append((_s["title"], _s["id"]))

    return show_titles


path_fixtures = Path(Path.cwd() / "tests/fixtures/")
input_file = Path(path_fixtures / "test_message_list.repr")

with input_file.open("r", encoding="utf-8") as f:
    data = ast.literal_eval(f.read())

show_titles = sonarr_series_titles()


co_code_hash = hashlib.sha256(
    marshal.dumps(match_to_show.__code__.co_code)
).hexdigest()[:24]

previous_file = Path(
    Path.cwd()
    / f"tests/fixtures/match_to_show/initial/match_to_show/func_{co_code_hash}.reprl"
)
if previous_file.exists():
    previous_file.unlink()


for message in data:
    creator = message.get("creator")
    title = message.get("title")
    datecode = message.get("datecode")

    main_title = f"{creator} :: {title}"

    _result = match_to_show(
        main_title,
        show_titles,
        stats_path=Path(path_fixtures / "match_to_show/initial"),
    )

    # print(_entry)
