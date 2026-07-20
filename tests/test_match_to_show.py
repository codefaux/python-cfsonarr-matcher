import ast
import hashlib
import marshal
import os
from pathlib import Path

import pytest
from cfsonarrmatcher import match_to_episode, match_to_show
from pyarr import SonarrAPI


@pytest.fixture
def sonarr_series_titles() -> list[tuple[str, int]]:
    show_titles: list[tuple[str, int]] = []
    with open("tests/fixtures/test_sonarr_series_titles.repr") as f:
        for _s in ast.literal_eval(f.read()):
            _t: tuple[str, int] = _s[0], _s[1]
            show_titles.append(_t)
    return show_titles


@pytest.fixture
def fixture_stats():
    with open("tests/fixtures/match_to_show/expected_stats.reprl") as f:
        return ast.literal_eval(f.read())


@pytest.fixture
def fixture_messages():
    with open("tests/fixtures/test_message_list.repr") as f:
        return ast.literal_eval(f.read())


@pytest.fixture
def fixture_messages_trouble_set():
    with open("tests/fixtures/test_message_trouble_set.repr") as f:
        return ast.literal_eval(f.read())


def normalize(value: dict) -> dict:
    """Return a normalized copy suitable for comparison."""

    if isinstance(value, dict):
        return {key: normalize(val) for key, val in value.items()}

    if isinstance(value, set):
        normalized = [normalize(item) for item in value]

        try:
            return sorted(normalized)
        except TypeError:
            # Mixed or non-comparable values; preserve original order
            return normalized

    if isinstance(value, list):
        normalized = [normalize(item) for item in value]

        try:
            return sorted(normalized)
        except TypeError:
            # Mixed or non-comparable values; preserve original order
            return normalized

    return value


def diff_dicts(a: dict, b: dict, path: str = "") -> list[str]:
    """
    Compare dict A against dict B.

    - Iterates only keys from A.
    - Ignores extra keys in B.
    - Reports missing keys in B.
    - Reports differing values.
    - Supports nested dictionaries.
    """
    a = normalize(a)
    b = normalize(b)

    differences = []

    for key, value_a in a.items():
        current_path: str = f"{path}.{key}" if path else str(key)

        if key not in b:
            differences.append(f"Missing key in B: {current_path}")
            continue

        value_b = b[key]

        if isinstance(value_a, dict) and isinstance(value_b, dict):
            differences.extend(diff_dicts(value_a, value_b, current_path))

        elif value_a != value_b:
            differences.append(
                f"Different value at {current_path}: "
                f"\n\tA={value_a!r}\n\tB={value_b!r}"
            )

    return differences


def test_match_to_show(
    tmp_path,
    sonarr_series_titles: list[tuple[str, int]],
    fixture_messages,
    fixture_stats,
):
    # sonarr_series_titles - list[str]
    # fixture_stats - list[dict[str, Any]] of past calls

    # extract inputs: list[str] from fixture_stats
    # for input in inputs:
    # - run match_to_show() with stats_path to write new stats
    # - compare new stats against relevant fixture_stats

    # iterate input titles from message list
    for message in fixture_messages:
        main_title = f"{message.get('creator', '')} :: {message.get('title', '')}"
        _ = match_to_show(
            input_title=main_title,
            sonarr_shows=sonarr_series_titles,
            stats_path=tmp_path,
        )

    co_code_hash = hashlib.sha256(
        marshal.dumps(match_to_show.__code__.co_code)
    ).hexdigest()[:24]

    output_file = Path(tmp_path) / f"match_to_show/func_{co_code_hash}.reprl"
    actual_stats: list[dict] = []

    with output_file.open() as f:
        for _line in f:
            actual_stats.append(ast.literal_eval(_line))

    all_differences = []

    for i, (actual, expected_item) in enumerate(zip(actual_stats, fixture_stats)):
        del expected_item["__func_sum"]
        differences = diff_dicts(expected_item, actual)

        if differences:
            all_differences.append(
                f"Index {i} ({expected_item["input_title"]}):\n"
                + "\n".join(differences)
            )

    assert not all_differences, "\n\n".join(all_differences)


def test_match_to_episode_trouble_set(
    tmp_path,
    sonarr_series_titles: list[tuple[str, int]],
    fixture_messages_trouble_set,
    fixture_stats,
):

    sonarr = SonarrAPI(os.getenv("SONARR_URL"), os.getenv("SONARR_API"))

    for match_test in fixture_messages_trouble_set:
        main_title = f"{match_test.get('input', {}).get('creator', '')} :: {match_test.get('input', {}).get('title', '')}"
        show_result = match_to_show(
            input_title=main_title,
            sonarr_shows=sonarr_series_titles,
            stats_path=tmp_path,
        )

        score, reason, id, show_name = show_result.get("best_results", {})[0]

        show_data = []
        for _ep in sonarr.get_episode(id, True):
            _ep["series"] = show_name
            _tag = ""

            show_data.append(
                {
                    "has_file": _ep["hasFile"],
                    "series": _ep["series"],
                    "series_id": _ep["seriesId"],
                    "season": _ep["seasonNumber"],
                    "episode": _ep["episodeNumber"],
                    "episode_id": _ep["id"],
                    "tag": _tag,
                    "title": _ep["title"],
                    "air_date": _ep.get("airDate", ""),
                    "air_date_utc": _ep.get("airDateUtc", ""),
                }
            )

        episode_result = match_to_episode(
            main_title,
            match_test.get("input", {}).get("datecode", ""),
            show_data,
            None,
            show_name,
        )

        diff = diff_dicts(match_test["expect"], episode_result)
        assert diff == []

        print(episode_result)
