import hashlib
import inspect
import marshal
from pathlib import Path
from types import FrameType


def write_stats(stats_path: Path | None = None):
    # dump format: {stats_path} / func_name / func_sum.reprl
    try:
        frame: FrameType | None = inspect.currentframe()
        if frame and frame.f_back:
            record: dict = frame.f_back.f_locals.copy()

            if stats_path is None:
                stats_path = record.get("stats_path", None)
                if not isinstance(stats_path, Path):
                    return

            schema = "|".join(sorted(record.keys()))
            record["__schema_sum"] = hashlib.sha256(schema.encode("ascii")).hexdigest()[
                :32
            ]
            record["__func_sum"] = hashlib.sha256(
                marshal.dumps(frame.f_back.f_code.co_code)
            ).hexdigest()[:32]

            del record["stats_path"]
            del record["executor"]
            del record["cand_scores"]
            del record["_cand"]

            filename = (
                stats_path
                / frame.f_back.f_code.co_name
                / f"func_{record["__func_sum"][:24]}.reprl"
            )

            filename.parent.mkdir(parents=True, exist_ok=True)

            with open(
                Path(filename),
                "a",
            ) as f:
                f.write(repr(record))
                f.write("\n")
    finally:
        del frame  # pyright: ignore[reportPossiblyUnboundVariable]
