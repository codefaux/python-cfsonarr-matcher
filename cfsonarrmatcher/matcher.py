import re
from collections import Counter
from datetime import date, datetime
from typing import Dict, List, Tuple

from dateutil import parser as dateparser
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


def extract_episode_hint(title: str) -> Tuple[int, int, str]:
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
            sub = results.get("sub") or ""
            return s_hint, e_hint, sub
    return -1, -1, ""


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


def score_candidate(
    main_title: str,
    season: int,
    episode: int,
    candidate_pool: Dict,
    candidate_token_freq: Dict[str, int],
    input_token_freq: Dict[str, int] | None = None,
    series_title: str | None = None,
) -> Tuple[int, str]:
    score = 0
    reasons = []

    if season != -1 or episode != -1:
        if candidate_pool["season"] == season and candidate_pool["episode"] == episode:
            score += 50
            reasons.append("season+ep exact fit: 50; ")
        elif candidate_pool["season"] == season or candidate_pool["episode"] == episode:
            score += 25
            reasons.append("season or ep matched: 25; ")
        else:
            score -= 5
            reasons.append("season/ep hint not unmatched: -5; ")

    input_tokens = set(fuzzutils.default_process(main_title).split())
    candidate_tokens = set(fuzzutils.default_process(candidate_pool["title"]).split())
    series_title_tokens = set(fuzzutils.default_process(series_title or "").split())

    token_score = int(fuzz.token_set_ratio(main_title, candidate_pool["title"]) * 0.25)
    weighted_recall = int(
        compute_weighted_overlap(
            input_tokens, candidate_tokens, candidate_token_freq, input_token_freq
        )
        * 50
    )

    score += token_score
    score += weighted_recall

    # Penalize missed tokens (input expected but not found)
    missed_tokens = (input_tokens - series_title_tokens) - (
        candidate_tokens - series_title_tokens
    )
    missed_penalty = len(missed_tokens) * 3
    score -= missed_penalty
    reasons.append(f"missed tokens: {len(missed_tokens)} (-{missed_penalty}); ")

    # Penalize extra tokens (unexpected tokens in candidate)
    extra_tokens = (candidate_tokens - series_title_tokens) - (
        input_tokens - series_title_tokens
    )
    extra_penalty = len(extra_tokens) * 2
    score -= int(extra_penalty)
    reasons.append(f"extra tokens: {len(extra_tokens)} (-{int(extra_penalty)}); ")

    reasons.append(f"token set similarity: {token_score}; ")
    reasons.append(f"weighted keyword recall: {weighted_recall}; ")

    return score, "".join(reasons)


def clean_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]+", " ", unidecode(text.lower()))


def deep_strip_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", unidecode(text.lower()))


def clean_sonarr_data(sonarr_data: list[dict]) -> list[dict]:
    return [
        {
            **entry,
            "series": clean_text(entry.get("series", "")),
            "title": clean_text(entry.get("title", "")),
            "orig_title": entry.get("title", ""),
        }
        for entry in sonarr_data
    ]


def match_title_to_sonarr_episode(
    main_title: str,
    airdate: str,
    sonarr_data: List[Dict],
    input_titles: List[str] | None = None,
    input_series: str | None = None,
) -> Dict:
    """Attempts to match a streaming title to a Sonarr entry with weighted keyword and date proximity scoring."""
    _mtd = unidecode(main_title.lower())
    _isd = unidecode((input_series or "").lower())

    _cmtd = clean_text(_mtd)
    cleaned_candidate_data = clean_sonarr_data(sonarr_data)

    # print(sonarr_data)

    cleaned_candidate_titles = [
        in_t.get("title") or "" for in_t in cleaned_candidate_data
    ]
    cleaned_others_titles = [clean_text(in_t) or "" for in_t in (input_titles or [])]

    candidate_token_freq = build_token_frequencies(cleaned_candidate_titles)
    input_token_freq = build_token_frequencies(cleaned_others_titles)
    season, episode, _ = extract_episode_hint(_mtd)

    best_match = None
    best_score = -1
    best_reason = ""

    for candidate in cleaned_candidate_data:
        _ctd = unidecode(candidate.get("title", "").lower())
        _cotd = unidecode(candidate.get("orig_title", "").lower())

        reason = f"match: '{_mtd}'  candidate: '{_cotd}';  \n\t"

        score, newreason = score_candidate(
            _cmtd,
            season,
            episode,
            candidate,
            candidate_token_freq,
            input_token_freq,
            input_series,
        )
        reason += newreason

        episode_date = candidate.get("air_date", "")
        episode_date_utc = candidate.get("air_date_utc", "")

        if episode_date != "":
            date_score_bonus = time_distance_score(airdate, episode_date_utc)
            if date_score_bonus > 0:
                date_gap = date_distance_days(airdate, episode_date_utc)
                score += date_score_bonus
                reason += f"date_gap={date_gap}d (bonus={date_score_bonus:.2f}); "
            else:
                reason += "no airdate bonus; "

        if _cotd in _mtd:
            reason += "verbatim match; "
            score += 50
        else:
            _len_ctd = len(_ctd)
            _len_cotd = len(_cotd)

            _fuzz_cotd = fuzzutils.default_process(_cotd)
            _fuzz_mtd = fuzzutils.default_process(_mtd)
            _fuzz_isd = fuzzutils.default_process(_isd)

            _deep_ctd = deep_strip_text(_ctd)
            _deep_mtd = deep_strip_text(_mtd)

            _hint_ctd = extract_episode_hint(_ctd)
            _hint_mtd = extract_episode_hint(_mtd)

            if _len_ctd < 5:
                reason += "short candidate, no verbatim match; "
                score -= 35 + (5 - _len_ctd) * 5

            if clean_text(_cotd) in clean_text(_mtd):
                reason += "cleaned verbatim match; "
                score += 40
            elif _fuzz_cotd in _fuzz_mtd:
                reason += "fuzzy match; "
                score += 25 + _len_cotd
            elif len(_isd) > 5 and (_fuzz_mtd.replace(_fuzz_isd, "")) in _fuzz_cotd:
                reason += "fuzzy match (-show); "
                score += 25 + _len_cotd
            elif _deep_ctd == _deep_mtd:
                reason += "deep strip match; "
                score += 25 + _len_cotd
            elif _deep_ctd in _deep_mtd:
                reason += "deep strip submatch; "
                score += 10 + _len_cotd

            if (
                len(_hint_ctd[2])
                and len(_hint_mtd[2])
                and _hint_ctd[:1] == _hint_mtd[:1]
            ):
                reason += "hint fingerprint match"
                score += 10

        if candidate.get("monitored", False) is True:
            score += 1
            reason += "monitored episode; "

        if candidate.get("hasFile", True) is False:
            score += 1
            reason += "no file; "

        if score > best_score:
            best_match = candidate
            best_score = score
            best_reason = reason

    return {
        "input": _mtd,
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
