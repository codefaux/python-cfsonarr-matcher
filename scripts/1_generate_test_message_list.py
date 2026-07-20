import ast
from pathlib import Path

input_file = Path(Path.cwd() / "tests/fixtures/input/dump.json")
output_file = Path(Path.cwd() / "tests/fixtures/test_message_list.repr")

output_data: list[dict[str, str]] = []

with input_file.open("r", encoding="utf-8") as f:
    data = ast.literal_eval(f.read())


for message in data["messages"]:
    if "text" in message and message["text"]:
        creator, datecode, title = [
            field.strip() for field in message["text"][0].split("::", 2)
        ]

        _entry: dict[str, str] = {
            "creator": creator,
            "datecode": datecode,
            "title": title,
        }

        print(_entry)

        output_data.append(_entry)

if output_file.exists():
    output_file.unlink()

with output_file.open("w", encoding="utf-8") as f:
    f.write(repr(output_data))
    f.write("\n")

print(f"-- {output_file.relative_to(Path.cwd())} updated.")
