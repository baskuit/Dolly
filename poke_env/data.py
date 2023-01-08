# -*- coding: utf-8 -*-
"""This module contains constant values used in the repository.
"""

import orjson  # pyre-ignore
import os

from functools import lru_cache
from typing import Any, Union
from typing import Dict


UNKNOWN_ITEM = "unknown_item"


def _compute_type_chart(chart_path: str) -> Dict[str, Dict[str, float]]:
    """Returns the pokemon type_chart.

    Returns a dictionnary representing the Pokemon type chart, loaded from a json file
    in `data`. This dictionnary is computed in this file as `TYPE_CHART`.

    :return: The pokemon type chart
    :rtype: Dict[str, Dict[str, float]]
    """
    with open(chart_path) as chart:
        json_chart = orjson.loads(chart.read())

    types = [str(entry["name"]).upper() for entry in json_chart]

    type_chart = {type_1: {type_2: 1.0 for type_2 in types} for type_1 in types}

    for entry in json_chart:
        type_ = entry["name"].upper()
        for immunity in entry["immunes"]:
            type_chart[type_][immunity.upper()] = 0.0
        for weakness in entry["weaknesses"]:
            type_chart[type_][weakness.upper()] = 0.5
        for strength in entry["strengths"]:
            type_chart[type_][strength.upper()] = 2.0

    return type_chart


@lru_cache(2**13)
def to_id_str(name: str) -> str:
    """Converts a full-name to its corresponding id string.
    :param name: The name to convert.
    :type name: str
    :return: The corresponding id string.
    :rtype: str
    """
    return "".join(char for char in name if char.isalnum()).lower()


POKEDEX: Dict[str, Any] = {}

with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "pokedex.json",
    )
) as pokedex:
    POKEDEX = orjson.loads(pokedex.read())

_missing_dex: Dict[str, Any] = {}
for key, value in POKEDEX.items():
    if "cosmeticFormes" in value:
        for other_form in value["cosmeticFormes"]:
            _missing_dex[to_id_str(other_form)] = value

# Alternative pikachu gmax forms
for name, value in POKEDEX.items():
    if name.startswith("pikachu") and name not in {
        "pikachu",
        "pikachugmax",
    }:
        _missing_dex[name + "gmax"] = POKEDEX["pikachugmax"]

POKEDEX.update(_missing_dex)

for name, value in POKEDEX.items():
    if "baseSpecies" in value:
        value["species"] = value["baseSpecies"]
    else:
        value["baseSpecies"] = to_id_str(name)


GEN4_POKEDEX: Dict[str, Any] = {}

with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "pokedex_by_gen",
        "gen4_pokedex.json",
    )
) as pokedex:
    GEN4_POKEDEX = orjson.loads(pokedex.read())

_missing_g4_dex: Dict[str, Any] = {}
for key, value in GEN4_POKEDEX.items():
    if "cosmeticFormes" in value:
        for other_form in value["cosmeticFormes"]:
            _missing_g4_dex[to_id_str(other_form)] = value

GEN3_POKEDEX: Dict[str, Any] = {}

with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "pokedex_by_gen",
        "gen3_pokedex.json",
    )
) as pokedex:
    GEN3_POKEDEX = orjson.loads(pokedex.read())

GEN3_POKEDEX = {
    key: value
    for key, value in GEN3_POKEDEX.items()
    if (0 < value["num"] <= 386) and ("baseSpecies" not in value or value["num"] == 386)
}

GEN4_POKEDEX.update(_missing_g4_dex)

for name, value in GEN4_POKEDEX.items():
    if "baseSpecies" in value:
        value["species"] = value["baseSpecies"]
    else:
        value["baseSpecies"] = to_id_str(name)


GEN5_POKEDEX: Dict[str, Any] = {}

with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "pokedex_by_gen",
        "gen5_pokedex.json",
    )
) as pokedex:
    GEN5_POKEDEX = orjson.loads(pokedex.read())

_missing_g5_dex: Dict[str, Any] = {}
for key, value in GEN5_POKEDEX.items():
    if "cosmeticFormes" in value:
        for other_form in value["cosmeticFormes"]:
            _missing_g5_dex[to_id_str(other_form)] = value

GEN5_POKEDEX.update(_missing_g5_dex)

for name, value in GEN5_POKEDEX.items():
    if "baseSpecies" in value:
        value["species"] = value["baseSpecies"]
    else:
        value["baseSpecies"] = to_id_str(name)


GEN6_POKEDEX: Dict[str, Any] = {}

with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "pokedex_by_gen",
        "gen6_pokedex.json",
    )
) as pokedex:
    GEN6_POKEDEX = orjson.loads(pokedex.read())

_missing_g6_dex: Dict[str, Any] = {}
for key, value in GEN6_POKEDEX.items():
    if "cosmeticFormes" in value:
        for other_form in value["cosmeticFormes"]:
            _missing_g6_dex[to_id_str(other_form)] = value

GEN6_POKEDEX.update(_missing_g6_dex)

for name, value in GEN6_POKEDEX.items():
    if "baseSpecies" in value:
        value["species"] = value["baseSpecies"]
    else:
        value["baseSpecies"] = to_id_str(name)


GEN7_POKEDEX: Dict[str, Any] = {}

with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "pokedex_by_gen",
        "gen7_pokedex.json",
    )
) as pokedex:
    GEN7_POKEDEX = orjson.loads(pokedex.read())

_missing_g7_dex: Dict[str, Any] = {}
for key, value in GEN7_POKEDEX.items():
    if "cosmeticFormes" in value:
        for other_form in value["cosmeticFormes"]:
            _missing_g7_dex[to_id_str(other_form)] = value

GEN7_POKEDEX.update(_missing_g7_dex)

for name, value in GEN7_POKEDEX.items():
    if "baseSpecies" in value:
        value["species"] = value["baseSpecies"]
    else:
        value["baseSpecies"] = to_id_str(name)


GEN8_POKEDEX: Dict[str, Any] = {}

with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "pokedex_by_gen",
        "gen8_pokedex.json",
    )
) as pokedex:
    GEN8_POKEDEX = orjson.loads(pokedex.read())

_missing_g8_dex: Dict[str, Any] = {}
for key, value in GEN8_POKEDEX.items():
    if "cosmeticFormes" in value:
        for other_form in value["cosmeticFormes"]:
            _missing_g8_dex[to_id_str(other_form)] = value

GEN8_POKEDEX.update(_missing_g8_dex)

for name, value in GEN8_POKEDEX.items():
    if "baseSpecies" in value:
        value["species"] = value["baseSpecies"]
    else:
        value["baseSpecies"] = to_id_str(name)


GEN_TO_POKEDEX: Dict[int, Dict[str, Any]] = {
    3: GEN3_POKEDEX,
    4: GEN4_POKEDEX,
    5: GEN5_POKEDEX,
    6: GEN6_POKEDEX,
    7: GEN7_POKEDEX,
    8: GEN8_POKEDEX,
}


MOVES: Dict[str, Any] = {}

with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "moves.json",
    )
) as moves:
    MOVES = orjson.loads(moves.read())

GEN3_MOVES: Dict[str, Any] = {}

with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "moves_by_gen",
        "gen3_moves.json",
    )
) as moves:
    GEN3_MOVES = orjson.loads(moves.read())

GEN3_MOVES = {
    key: value
    for key, value in GEN3_MOVES.items()
    if ("isNonstandard" not in value or value["isNonstandard"] is None)
}

GEN4_MOVES: Dict[str, Any] = {}

with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "moves_by_gen",
        "gen4_moves.json",
    )
) as moves:
    GEN4_MOVES = orjson.loads(moves.read())


GEN5_MOVES: Dict[str, Any] = {}

with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "moves_by_gen",
        "gen5_moves.json",
    )
) as moves:
    GEN5_MOVES = orjson.loads(moves.read())


GEN6_MOVES: Dict[str, Any] = {}

with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "moves_by_gen",
        "gen6_moves.json",
    )
) as moves:
    GEN6_MOVES = orjson.loads(moves.read())


GEN7_MOVES: Dict[str, Any] = {}

with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "moves_by_gen",
        "gen7_moves.json",
    )
) as moves:
    GEN7_MOVES = orjson.loads(moves.read())


GEN8_MOVES: Dict[str, Any] = {}

with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "moves_by_gen",
        "gen8_moves.json",
    )
) as moves:
    GEN8_MOVES = orjson.loads(moves.read())


GEN_TO_MOVES: Dict[int, Dict[str, Any]] = {
    3: GEN3_MOVES,
    4: GEN4_MOVES,
    5: GEN5_MOVES,
    6: GEN6_MOVES,
    7: GEN7_MOVES,
    8: GEN8_MOVES,
}

NATURES: Dict[str, Dict[str, Union[int, float]]] = {}

with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "natures.json",
    )
) as natures:
    NATURES = orjson.loads(natures.read())


with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "replay_template.html",
    )
) as f:
    REPLAY_TEMPLATE = f.read()


with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "items.json",
    )
) as f:
    ITEMS = orjson.loads(f.read())


with open(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "data",
        "abilities.json",
    )
) as f:
    ABILITIES = orjson.loads(f.read())


def str_to_id(datum, str):
    id = list(datum).index(str)
    return id


POKEDEX = GEN3_POKEDEX
MOVES = GEN3_MOVES

POKEDEX_STOI = {name: index + 1 for index, name in enumerate(sorted(POKEDEX.keys()))}
POKEDEX_ITOS = {index: action for action, index in POKEDEX_STOI.items()}
ABILITIES_STOI = {
    name: index + 1 for index, name in enumerate(sorted(ABILITIES.keys()))
}
ITEMS_STOI = {name: index + 1 for index, name in enumerate(sorted(ITEMS.keys()))}
MOVES_STOI = {name: index + 1 for index, name in enumerate(sorted(MOVES.keys()))}
MOVES_ITOS = {index: action for action, index in MOVES_STOI.items()}

MOVE_CUTOFF = len(MOVES_STOI)
POKEDEX_CUTOFF = len(POKEDEX_STOI)

ACTIONS_STOI = {}
for key, value in MOVES_STOI.items():
    ACTIONS_STOI[key] = value - 1

for key, value in POKEDEX_STOI.items():
    ACTIONS_STOI[key] = value - 1 + MOVE_CUTOFF

ACTIONS_ITOS = {index: action for action, index in ACTIONS_STOI.items()}

ACTION_SPACE = len(ACTIONS_STOI) + 1
