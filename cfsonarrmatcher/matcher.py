import re
from collections import Counter
from datetime import date, datetime
from typing import Dict, List, Tuple

from dateutil import parser as dateparser
from rapidfuzz import fuzz
from rapidfuzz import utils as fuzzutils


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


def extract_episode_hint(title: str) -> Tuple[int, int]:
    """Attempts to parse season and episode numbers from the title."""
    patterns = [
        r"S(?P<season_hint>\d+)[\W_]-[\W_]E?(?P<episode_hint>\d+)",  # S5 - 6
        r"S(?P<season_hint>\d+)E(?P<episode_hint>\d+)",  # S2E3
        r"Season[^\d]*(?P<season_hint>\d+)[^\d]+Episode[^\d]*(?P<episode_hint>\d+)",  # Season 2 Episode 3
        r"S(?P<season_hint>\d+)[^\d]+Ep(?:isode)?[^\d]*(?P<episode_hint>\d+)",  # S2 Ep 3
        r"Episode[^\d]*(?P<episode_hint>\d+)",  # Episode 3
        r"Ep[^\d]*(?P<episode_hint>\d+)",  # Ep 3
        r"^[^\d]+S(?:eason)?[^\d]?(?P<season_hint>\d+)[^\d]*\(?E?\d+-E?\d+\)?",  # S5 (01-12)  | Season 3 (E01-12) | Season4 (e01-E12)
    ]
    for pattern in patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            results = match.groupdict()
            hint_s = int(results.get("season_hint") or -1)
            hint_e = int(results.get("episode_hint") or -1)
            return hint_s, hint_e
    return -1, -1


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
            reasons.append("season+ep exact fit: 50")
        elif candidate_pool["season"] == season or candidate_pool["episode"] == episode:
            score += 25
            reasons.append("season or ep matched: 25")
        else:
            score -= 5
            reasons.append("season/ep but unmatched: -5")

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
    reasons.append(f"missed tokens: {len(missed_tokens)} (-{missed_penalty})")

    # Penalize extra tokens (unexpected tokens in candidate)
    extra_tokens = (candidate_tokens - series_title_tokens) - (
        input_tokens - series_title_tokens
    )
    extra_penalty = len(extra_tokens) * 2
    score -= int(extra_penalty)
    reasons.append(f"extra tokens: {len(extra_tokens)} (-{int(extra_penalty)})")

    reasons.append(f"token set similarity: {token_score}")
    reasons.append(f"weighted keyword recall: {weighted_recall}")

    return score, "; ".join(reasons)


def clean_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text).lower()


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
    cleaned_title = clean_text(main_title)
    cleaned_candidate_data = clean_sonarr_data(sonarr_data)

    # print(sonarr_data)

    cleaned_candidate_titles = [
        in_t.get("title") or "" for in_t in cleaned_candidate_data
    ]
    cleaned_others_titles = [clean_text(in_t) or "" for in_t in (input_titles or [])]

    candidate_token_freq = build_token_frequencies(cleaned_candidate_titles)
    input_token_freq = build_token_frequencies(cleaned_others_titles)
    season, episode = extract_episode_hint(main_title)

    best_match = None
    best_score = -1
    best_reason = ""

    for candidate in cleaned_candidate_data:
        score, reason = score_candidate(
            cleaned_title,
            season,
            episode,
            candidate,
            candidate_token_freq,
            input_token_freq,
            input_series,
        )

        # Date distance bonus
        episode_date = candidate.get("air_date", "")
        episode_date_utc = candidate.get("air_date_utc", "")
        if episode_date != "":
            date_score_bonus = time_distance_score(airdate, episode_date_utc)
            if date_score_bonus > 0:
                date_gap = date_distance_days(airdate, episode_date_utc)
                score += date_score_bonus
                reason += f"; date_gap={date_gap}d (bonus={date_score_bonus:.2f})"
            else:
                reason += "; no airdate bonus"

        if candidate.get("monitored", False) is True:
            score += 1
            reason += "; monitored episode"

        if candidate.get("hasFile", True) is False:
            score += 1
            reason += "; no file"

        if score > best_score:
            best_match = candidate
            best_score = score
            best_reason = reason

    return {
        "input": main_title,
        "matched_show": best_match["series"] if best_match else None,
        "season": best_match["season"] if best_match else None,
        "episode": best_match["episode"] if best_match else None,
        "episode_title": best_match["title"] if best_match else None,
        "episode_orig_title": best_match["orig_title"] if best_match else None,
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
        processed_show = fuzzutils.default_process(show_title)
        show_tokens = set(processed_show.split())

        # Priority boost if show name appears verbatim
        verbatim_match = processed_show in fuzzutils.default_process(main_title)
        verbatim_bonus = 35 + len(show_title) if verbatim_match else 0

        # Token similarity and keyword overlap
        token_score = fuzz.token_set_ratio(main_title, show_title)
        keyword_overlap = (
            len(show_tokens & input_tokens) / len(show_tokens) if show_tokens else 0
        )

        score = verbatim_bonus + int(token_score * 0.10) + int(keyword_overlap * 50)

        reason = (
            f"{'verbatim match; ' if verbatim_match else ''}"
            f"token set similarity: {token_score}%, "
            f"keyword overlap: {int(keyword_overlap * 100)}%"
        )

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
