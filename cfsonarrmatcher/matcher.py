import re
from collections import Counter
from datetime import date, datetime
from typing import Dict, List, Tuple

from dateutil import parser as dateparser
from fauxjson import persist_wrap as _dump
from rapidfuzz import fuzz
from rapidfuzz import utils as fuzzutils
from unidecode import unidecode


def parse_date(date_input: str | date) -> datetime | None:
    if isinstance(date_input, date):
        date_input = str(date_input)

    try:
        return dateparser.parse(date_input, fuzzy=True)
    except (ValueError, TypeError):
        return None


def date_distance_days(date1_input: str | date, date2_input: str | date) -> int:
    date1 = parse_date(date1_input)
    date2 = parse_date(date2_input)
    if date1 is None or date2 is None:
        return -1
    return abs((date1.date() - date2.date()).days)


def time_distance_score(
    datetime1_input: str | datetime, datetime2_input: str | datetime
) -> int:
    from datetime import timezone

    max_hours_limit = 72
    max_score = 80
    decay_power = 2.4

    datetime1 = parse_date(datetime1_input)
    datetime2 = parse_date(datetime2_input)
    if datetime1:
        datetime1 = datetime1.replace(tzinfo=timezone.utc)
    if datetime2:
        datetime2 = datetime2.replace(tzinfo=timezone.utc)

    if datetime1 is None or datetime2 is None:
        return -1

    distance_hours = abs((datetime1 - datetime2).total_seconds()) / 3600

    if distance_hours > max_hours_limit:
        return 0

    normalized = distance_hours / max_hours_limit
    score = (1 - normalized**decay_power) * max_score

    return int(score)


def extract_episode_hint(title: str) -> Tuple[int, int, str | None]:
    """Attempts to parse season and episode numbers from the title."""
    patterns = [
        r"(?=S(?:eason)?[\W_]?(?P<_s>\d{1,4}(?!\d))|(?(_s)E?|E)p?(?:isode)?[\W_]?\d{1,4}(?!\d))(?P<sub>(?:S(?:eason)?[\W_]?(?P<s_hint>\d{,4}))?[^a-zA-Z0-9]{,5}(?:(?(_s)E?|E)p?(?:isode)?[\W_]?(?P<e_hint>\d{1,4}(?!\w)))?)"
        # Below should be redundant to above, now.
        # r"(?P<substring>S(?P<season_hint>\d+)[\W_]?-[\W_]?E?(?P<episode_hint>\d+))",  # S5 - 6
        # r"S(?P<season_hint>\d+)E(?P<episode_hint>\d+)",  # S2E3
        # r"Season[^\d]*(?P<season_hint>\d+)[^\d]+Episode[^\d]*(?P<episode_hint>\d+)",  # Season 2 Episode 3
        # r"S(?P<season_hint>\d+)[^\d]+Ep(?:isode)?[^\d]*(?P<episode_hint>\d+)",  # S2 Ep 3
        # r"Episode[^\d]*(?P<episode_hint>\d+)",  # Episode 3
        # r"Ep[^\d]*(?P<episode_hint>\d+)",  # Ep 3
        # r"^[^\d]+S(?:eason)?[^\d]?(?P<season_hint>\d+)[^\d]*\(?E?\d+-E?\d+\)?",  # S5 (01-12)  | Season 3 (E01-12) | Season4 (e01-E12)
    ]
    for pattern in patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            results = match.groupdict()
            s_hint = int(results.get("s_hint") or -1)
            e_hint = int(results.get("e_hint") or -1)
            sub = results.get("sub")
            return s_hint, e_hint, sub
    return -1, -1, None


def build_token_frequencies(token_pool: List[str]) -> Dict[str, int]:
    token_counts = Counter()
    for entry in token_pool:
        tokens = fuzzutils.default_process(entry).split()
        token_counts.update(tokens)
    return token_counts


def inv_freq(freq: int) -> float:
    import math

    return 1 / math.log1p(freq)


def normalize_token(token: str) -> str:
    # Strip leading zeros for purely numeric tokens
    if token.isdigit():
        return str(int(token))
    return token


def compute_weighted_overlap(
    input_tokens: set,
    candidate_tokens: set,
    candidate_freq_map: Dict[str, int],
    input_freq_map: Dict[str, int] | None = None,
) -> float:
    if not candidate_tokens:
        return 0.0

    normalized_input_tokens = {normalize_token(t): t for t in input_tokens}

    total_weight = 0.0
    overlap_weight = 0.0

    for token in candidate_tokens:
        cand_weight = inv_freq(candidate_freq_map.get(token, 1))
        total_weight += cand_weight

        norm_token = normalize_token(token)

        if norm_token in normalized_input_tokens:
            matched_input_token = normalized_input_tokens[norm_token]

            input_weight = (
                inv_freq(input_freq_map.get(matched_input_token, 1))
                if input_freq_map is not None
                else 1.0
            )
            overlap_weight += cand_weight * input_weight

    return overlap_weight / total_weight if total_weight > 0 else 0.0


def clean_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]+", " ", unidecode(text.lower()))


def deep_strip_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", unidecode(text.lower()))


def clean_sonarr_data(sonarr_data: list[dict]) -> list[dict]:
    return [
        {
            **entry,
            "series_c": clean_text(entry.get("series", "")),
            "title_c": clean_text(entry.get("title", "")),
            "orig_title": entry.get("title", ""),
        }
        for entry in sonarr_data
    ]


@_dump
def match_title_to_sonarr_episode(
    main_title: str,
    airdate: str,
    candidate_data: List[Dict],
    input_titles: List[str] | None = None,
    input_series: str | None = None,
) -> Dict:
    """Attempts to match a streaming title to a Sonarr entry with weighted keyword and date proximity scoring."""

    _main_title_d = unidecode(main_title.lower())
    _input_series_d = unidecode((input_series or "").lower())

    _main_title_c = clean_text(_main_title_d)
    _cand_data_c = clean_sonarr_data(candidate_data)

    _cand_titles_c = [__cand_c.get("title_c") or "" for __cand_c in _cand_data_c]
    _other_titles_c = [clean_text(__title) or "" for __title in (input_titles or [])]

    _cand_titles_c_tokenfreq = build_token_frequencies(_cand_titles_c)
    _other_titles_c_tokenfreq = build_token_frequencies(_other_titles_c)

    _hint_main_title_d = extract_episode_hint(_main_title_d)

    _main_title_season_hint, _main_title_episode_hint, _main_title_substr_hint = (
        _hint_main_title_d
    )

    best_match = None
    best_score = -1
    best_reason = ""

    for __cand_c in _cand_data_c:
        _cand_title_c = __cand_c.get("title_c", "")
        _cand_tag_d = unidecode(__cand_c.get("tag") or "")
        _cand_orig_title_d = unidecode(__cand_c.get("orig_title", "").lower())

        reason = f"input: '{_main_title_d}'  candidate: '{_cand_orig_title_d}';  \n\t"

        reason += f"HINT: {_main_title_season_hint, _main_title_episode_hint, _main_title_substr_hint}"

        score = 0

        if _main_title_season_hint != -1 or _main_title_episode_hint != -1:
            if (
                __cand_c["season"] == _main_title_season_hint
                and __cand_c["episode"] == _main_title_episode_hint
            ):
                score += 50
                reason += "season+ep exact fit: 50; "
            elif (
                __cand_c["season"] == _main_title_season_hint
                or __cand_c["episode"] == _main_title_episode_hint
            ):
                score += 20
                reason += "season or ep matched: 25; "
            else:
                _main_title_substr_hint = ""  # invalidate if irrelevant
                score -= 10
                reason += "season/ep hint not unmatched: -5; "

        _main_title_c_tokens = set(fuzzutils.default_process(_main_title_c).split())
        _input_series_d_tokens = set(fuzzutils.default_process(_input_series_d).split())
        _cand_title_c_tokens = set(fuzzutils.default_process(_cand_title_c).split())
        _input_tag_d_tokens = set(fuzzutils.default_process(_cand_tag_d).split())
        _input_hint_tokens = set(
            unidecode(clean_text(_main_title_substr_hint or "").lower()).split()
        )

        reason += f"TOKENS: {_input_hint_tokens}"

        token_score = int(fuzz.token_set_ratio(_main_title_c, __cand_c["title"]) * 0.25)
        weighted_recall = int(
            compute_weighted_overlap(
                _main_title_c_tokens,
                _cand_title_c_tokens,
                _cand_titles_c_tokenfreq,
                _other_titles_c_tokenfreq,
            )
            * 50
        )

        score += token_score
        score += weighted_recall

        reason += f"token set similarity: {token_score}; "
        reason += f"weighted keyword recall: {weighted_recall}; "

        episode_date = __cand_c.get("air_date", "")
        episode_date_utc = __cand_c.get("air_date_utc", "")

        if episode_date != "":
            date_score_bonus = time_distance_score(airdate, episode_date_utc)
            if date_score_bonus > 0:
                date_gap = date_distance_days(airdate, episode_date_utc)
                score += date_score_bonus
                reason += f"date_gap={date_gap}d (bonus={date_score_bonus:.2f}); "
            else:
                reason += "no airdate bonus; "

        if _cand_orig_title_d in _main_title_d:
            reason += "verbatim match; "
            score += 50
        else:
            _len_cand_title_c = len(_cand_title_c)
            _len_cand_orig_title_d = len(_cand_orig_title_d)

            _fuzz_cand_orig_title_d = fuzzutils.default_process(_cand_orig_title_d)
            _fuzz_main_title_d = fuzzutils.default_process(_main_title_d)
            _fuzz_input_series_d = fuzzutils.default_process(_input_series_d)

            _deep_cand_title_c = deep_strip_text(_cand_title_c)
            _deep_main_title_d = deep_strip_text(_main_title_d)

            _hint_cand_title_c = extract_episode_hint(_cand_title_c)

            if _len_cand_title_c < 5:
                reason += "short candidate, no verbatim match; "
                score -= 35 + (5 - _len_cand_title_c) * 5

            if clean_text(_cand_orig_title_d) in clean_text(_main_title_d):
                reason += "cleaned verbatim match; "
                score += 40
            elif _fuzz_cand_orig_title_d in _fuzz_main_title_d:
                reason += "fuzzy match; "
                score += 25 + _len_cand_orig_title_d
            elif (
                len(_input_series_d) > 5
                and (_fuzz_main_title_d.replace(_fuzz_input_series_d, ""))
                in _fuzz_cand_orig_title_d
            ):
                reason += "fuzzy match (-show); "
                score += 25 + _len_cand_orig_title_d
            elif _deep_cand_title_c == _deep_main_title_d:
                reason += "deep strip match; "
                score += 25 + _len_cand_orig_title_d
            elif _deep_cand_title_c in _deep_main_title_d:
                reason += "deep strip submatch; "
                score += 10 + _len_cand_orig_title_d

            print(f"Fingerprint: CTD: {_hint_cand_title_c}  MTD: {_hint_main_title_d}")
            if (
                len(_hint_cand_title_c[2] or "")
                and len(_hint_main_title_d[2] or "")
                and _hint_cand_title_c[:1] == _hint_main_title_d[:1]
            ):
                if _hint_cand_title_c[2] == _main_title_c:
                    reason += "verbatim match, candidate hint is title; "
                    score += 25
                else:
                    reason += "hint fingerprint match; "
                    score += 15

            # Penalize missed tokens (in candidate but not input)
            missed_tokens = (_cand_title_c_tokens - _input_series_d_tokens) - (
                _main_title_c_tokens - _input_series_d_tokens
            )

            reason += f"MISSED TOKENS: {missed_tokens}"

            missed_penalty = len(missed_tokens) * 3
            score -= missed_penalty
            reason += f"missed tokens: {len(missed_tokens)} (-{missed_penalty}); "

            # Penalize extra tokens (unexpected tokens in candidate)
            extra_tokens = (
                _main_title_c_tokens
                - _input_series_d_tokens
                - _input_tag_d_tokens
                - _input_hint_tokens
            ) - (_cand_title_c_tokens - _input_series_d_tokens)

            reason += f"EXTRA TOKENS: {extra_tokens}"

            extra_penalty = len(extra_tokens) * 3
            score -= int(extra_penalty)
            reason += f"extra tokens: {len(extra_tokens)} (-{int(extra_penalty)}); "

        if __cand_c.get("monitored", False) is True:
            score += 1
            reason += "monitored episode; "

        if __cand_c.get("hasFile", True) is False:
            score += 1
            reason += "no file; "

        if score > best_score:
            best_match = __cand_c
            best_score = score
            best_reason = reason

    return {
        "input": _main_title_d,
        "matched_show": best_match["series"] if best_match else None,
        "matched_series_id": best_match["series_id"] if best_match else None,
        "season": best_match["season"] if best_match else None,
        "episode": best_match["episode"] if best_match else None,
        "episode_title": best_match["title"] if best_match else None,
        "episode_orig_title": best_match["orig_title"] if best_match else None,
        "full_match": best_match if best_match else None,
        "score": best_score,
        "reason": best_reason,
    }


def match_title_to_sonarr_show(main_title: str, sonarr_shows) -> Dict:
    """Matches a streaming title to the best-matching Sonarr show using strict verbatim and token-based scoring."""
    input_tokens = set(fuzzutils.default_process(main_title).split())

    best_match = None
    best_score = -1
    best_reason = ""
    best_id = ""

    for show_title, show_id in set(sonarr_shows):
        score = 0
        reason = ""

        processed_show = fuzzutils.default_process(show_title)
        show_tokens = set(processed_show.split())

        if show_title.lower() in main_title.lower():
            reason += "verbatim match; "
            score += 75
        else:
            if len(show_title) < 5:
                reason += "short candidate, no verbatim match; "
                score -= 35 + (5 - len(show_title)) * 5
            if processed_show in fuzzutils.default_process(main_title):
                reason += "fuzzy match; "
                score += 35 + len(show_title)

        # Token similarity and keyword overlap
        token_score = fuzz.token_set_ratio(main_title, show_title)
        score += int(token_score * 0.10)
        reason += f"token set similarity: {token_score}; "

        keyword_overlap = (
            len(show_tokens & input_tokens) / len(show_tokens) if show_tokens else 0
        )
        score += int(keyword_overlap * 50)
        reason += f"keyword overlap: {keyword_overlap}; "

        score = max(0, score)

        if score > best_score:
            best_id = show_id
            best_match = show_title
            best_score = score
            best_reason = reason

    return {
        "input": main_title,
        "matched_id": best_id,
        "matched_show": best_match,
        "score": best_score,
        "reason": best_reason,
    }
