"""Template compression for repetitive, line-based text such as logs.

Why this exists: the dictionary + LZ77 compressor points back by character
distance, and a language model can't count characters. Here every repeated line
shape is written once as a template, and each line becomes a short row of values.
A model can read that the way it reads a table, with no decoding rules to learn.

Format::

    @1 {} {} INFO request_id={} method=GET path={} status={} duration_ms={}
    @2 {} {} ERROR upstream_timeout service=auth after_ms={} request_id={}
    ---
    1|2026-11-12|10:03:00|31244567|/api/v1/users/143|200|57
    2|2026-11-12|10:03:04|2210|31244567

Header lines start with ``@id``, then ``---``, then one row per input line, in
the original order. ``{}`` marks a value that changes between lines. A value that
is the same in every line of a template is written into the template itself.
A line that can't be handled safely is kept as ``!`` plus the raw line.
"""

from __future__ import annotations

SEPARATOR = "---"
SLOT = "{}"
VALUE_JOIN = "|"


def _shape(tokens: list[str]) -> tuple[tuple[str, str], ...] | None:
    """Describe a line as (kind, text) pairs, or None if it can't be templated."""
    shape = []
    for token in tokens:
        if SLOT in token or VALUE_JOIN in token:
            return None
        if "=" in token:
            key, _, value = token.partition("=")
            shape.append(("kv", key))
        elif any(ch.isdigit() for ch in token):
            shape.append(("var", ""))
        else:
            shape.append(("lit", token))
    return tuple(shape)


def _values(tokens: list[str], shape: tuple[tuple[str, str], ...]) -> list[str]:
    values = []
    for token, (kind, _text) in zip(tokens, shape):
        if kind == "kv":
            values.append(token.partition("=")[2])
        elif kind == "var":
            values.append(token)
    return values


def _template(shape: tuple[tuple[str, str], ...], constants: dict[int, str]) -> str:
    """Write the template line. `constants` maps slot index to an inlined value."""
    parts, slot = [], 0
    for kind, text in shape:
        if kind == "lit":
            parts.append(text)
        else:
            value = constants.get(slot)
            body = SLOT if value is None else value
            parts.append(f"{text}={body}" if kind == "kv" else body)
            slot += 1
    return " ".join(parts)


class StructuredCompressor:
    """Lossless template compression. `decompress(compress(x)) == x` for any text."""

    def compress(self, text: str) -> str:
        lines = text.split("\n")
        shapes: list[tuple | None] = []
        groups: dict[tuple, list[int]] = {}
        for index, line in enumerate(lines):
            tokens = line.split(" ")
            shape = _shape(tokens)
            shapes.append(shape)
            if shape is not None and any(kind != "lit" for kind, _ in shape):
                groups.setdefault(shape, []).append(index)

        group_ids: dict[tuple, int] = {}
        constants: dict[int, dict[int, str]] = {}
        headers = []
        for shape, members in groups.items():
            gid = len(group_ids) + 1
            group_ids[shape] = gid
            columns = list(zip(*[_values(lines[i].split(" "), shape) for i in members]))
            fixed = {slot: col[0] for slot, col in enumerate(columns) if len(members) > 1 and len(set(col)) == 1}
            constants[gid] = fixed
            headers.append(f"@{gid} {_template(shape, fixed)}")

        rows = []
        for index, line in enumerate(lines):
            shape = shapes[index]
            if shape is None or shape not in group_ids:
                rows.append("!" + line)
                continue
            gid = group_ids[shape]
            values = _values(line.split(" "), shape)
            kept = [v for slot, v in enumerate(values) if slot not in constants[gid]]
            rows.append(VALUE_JOIN.join([str(gid)] + kept))
        return "\n".join(headers + [SEPARATOR] + rows)

    def decompress(self, packed: str) -> str:
        lines = packed.split("\n")
        split_at = lines.index(SEPARATOR)
        templates = {}
        for header in lines[:split_at]:
            gid, _, template = header[1:].partition(" ")
            templates[int(gid)] = template
        out = []
        for row in lines[split_at + 1:]:
            if row.startswith("!"):
                out.append(row[1:])
                continue
            gid, *values = row.split(VALUE_JOIN)
            pieces = templates[int(gid)].split(SLOT)
            line = pieces[0]
            for value, piece in zip(values, pieces[1:]):
                line += value + piece
            out.append(line)
        return "\n".join(out)
