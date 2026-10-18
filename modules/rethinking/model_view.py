"""Teacher chat-template and EOS metadata for base-student OPD."""

import json
from pathlib import Path


def apply_teacher_metadata(student, teacher, destination, link_weights=False):
    from transformers import AutoTokenizer

    student, teacher, destination = map(Path, (student, teacher, destination))
    source = AutoTokenizer.from_pretrained(student, local_files_only=True)
    target = AutoTokenizer.from_pretrained(teacher, local_files_only=True)
    if source.get_vocab() != target.get_vocab():
        raise ValueError(
            "Teacher template requires identical student/teacher token IDs"
        )
    if not target.chat_template:
        raise ValueError("Teacher tokenizer has no chat template")
    destination.mkdir(parents=True, exist_ok=True)
    if link_weights:
        for path in student.iterdir():
            if path.is_file() and path.suffix in {
                ".json",
                ".safetensors",
                ".bin",
                ".txt",
                ".model",
                ".jinja",
            }:
                if path.name not in {
                    "tokenizer_config.json",
                    "generation_config.json",
                    "chat_template.jinja",
                }:
                    (destination / path.name).symlink_to(path.resolve())
    tokenizer_config = json.loads((student / "tokenizer_config.json").read_text())
    tokenizer_config["chat_template"] = target.chat_template
    (destination / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2) + "\n"
    )
    (destination / "chat_template.jinja").write_text(target.chat_template)
    generation_path = student / "generation_config.json"
    generation = (
        json.loads(generation_path.read_text()) if generation_path.exists() else {}
    )
    stops = set()
    for tokenizer in [source, target]:
        if tokenizer.eos_token_id is not None:
            stops.add(tokenizer.eos_token_id)
    generation.update(eos_token_id=sorted(stops), pad_token_id=source.pad_token_id)
    (destination / "generation_config.json").write_text(
        json.dumps(generation, indent=2) + "\n"
    )
