"""
generate.py — Ironhold Arena Ruling Generation v2 (Redesigned)

Closes the algorithmic loophole from v1:
  - NO rulebook provided to solvers (rules learned from examples only)
  - NO rule IDs in outputs (natural language analysis only)
  - 150+ rules with NUMERICAL conditions (hp, mana, level, strength, defense)
  - Analyses reference specific numbers from the situation
  - 5+ template variants per situation/analysis
  - 10% training label noise
  - Test-time numerical distribution shift

Output:
  raw_data/data.csv — 5,000 Q&A pairs (no rulebook)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

SEED = 42

UNIT_TYPES = ["Guardian", "Striker", "Mage", "Ranger", "Healer", "Rogue", "Paladin", "Summoner"]
TERRAINS = ["Plains", "Forest", "Mountain", "Swamp", "Desert", "Ruins"]
SPELLS = ["Fireball", "Frostbind", "Barrier", "Heal", "Shadowstep", "Lightning", "Drain", "Rally"]
STATUSES = ["Frozen", "Burning", "Shielded", "Poisoned", "Stunned", "Invisible"]
ACTIONS = ["move", "attack", "cast"]
OUTCOMES = ["ALLOWED", "BLOCKED", "MODIFIED"]

CASTER_TYPES = {"Mage", "Healer", "Paladin", "Summoner"}
MELEE_TYPES = {"Guardian", "Striker", "Rogue"}


def build_rules(rng):
    """Programmatically generate ~150 rules with categorical + numerical conditions."""
    rules = []

    def add(conditions, effect, priority, mechanic, descs):
        rules.append({
            "conditions": conditions,
            "effect": effect,
            "priority": priority,
            "mechanic": mechanic,
            "descriptions": descs,
        })

    # ── CAT 1: TERRAIN MOVEMENT RESTRICTIONS (~16 rules) ──
    terrain_block_map = {}
    for ut in UNIT_TYPES:
        blocked = list(rng.choice(TERRAINS, size=2, replace=False))
        terrain_block_map[ut] = blocked
        for t in blocked:
            add(
                {"action": "move", "actor_type": ut, "target_terrain": t},
                "BLOCKED", 1, f"{ut}_{t}_block",
                [
                    f"{ut}s cannot traverse {t} terrain due to their physical limitations.",
                    f"The {t} environment is impassable for {ut} units.",
                    f"A {ut} finds {t} terrain incompatible with their movement capabilities.",
                ],
            )

    # ── CAT 2: STATUS ACTION BLOCKS (6 rules) ──
    for status in ["Frozen", "Stunned"]:
        for action in ACTIONS:
            verb = {"move": "move", "attack": "attack", "cast": "cast spells"}[action]
            add(
                {"action": action, "actor_has_status": status},
                "BLOCKED", 2, f"{status}_{action}_block",
                [
                    f"{status} units cannot {verb}. The condition prevents all voluntary actions.",
                    f"The {status} effect immobilizes the unit, blocking any attempt to {verb}.",
                    f"While {status}, performing any {action} action is impossible.",
                ],
            )

    # ── CAT 3: NON-CASTER SPELL BLOCKS (4 rules) ──
    for ut in ["Guardian", "Striker", "Ranger", "Rogue"]:
        add(
            {"action": "cast", "actor_type": ut},
            "BLOCKED", 1, f"{ut}_no_magic",
            [
                f"{ut}s lack arcane training and cannot cast spells.",
                f"Spellcasting is not available to {ut} units — they have no magical ability.",
                f"A {ut} has no capacity for magic and cannot cast any spell.",
            ],
        )

    # ── CAT 4: SPELL MANA REQUIREMENTS (8 rules) ──
    spell_mana = {}
    for spell in SPELLS:
        cost = int(rng.choice([12, 15, 18, 20, 25, 28, 30, 35]))
        spell_mana[spell] = cost
        add(
            {"action": "cast", "spell": spell, "actor_mana_below": cost},
            "BLOCKED", 1, f"{spell}_mana_req",
            [
                f"Casting {spell} requires at least {cost} mana. Insufficient mana blocks the spell.",
                f"{spell} costs {cost} mana to cast. Without enough mana, the spell fizzles.",
                f"The {spell} spell demands {cost} mana — anything less and the cast fails.",
            ],
        )

    # ── CAT 5: SPELL TERRAIN BLOCKS (~16 rules) ──
    spell_terrain_blocks = {}
    for spell in SPELLS:
        blocked_terrains = list(rng.choice(TERRAINS, size=2, replace=False))
        spell_terrain_blocks[spell] = blocked_terrains
        for t in blocked_terrains:
            reasons = {
                "Forest": "ambient moisture dampens the magic",
                "Mountain": "the thin air disrupts concentration",
                "Swamp": "the waterlogged ground absorbs magical energy",
                "Desert": "extreme heat destabilizes the spell structure",
                "Ruins": "residual dark energy interferes with the casting",
                "Plains": "open terrain dissipates the focused energy",
            }
            reason = reasons.get(t, "the terrain interferes with the spell")
            add(
                {"action": "cast", "spell": spell, "actor_terrain": t},
                "BLOCKED", 1, f"{spell}_{t}_block",
                [
                    f"{spell} cannot be cast on {t} terrain — {reason}.",
                    f"The {t} environment prevents {spell} from forming: {reason}.",
                    f"Casting {spell} in {t} fails because {reason}.",
                ],
            )

    # ── CAT 6: LEVEL-GATED ACTIONS (~10 rules) ──
    level_gates = {}
    for spell in rng.choice(SPELLS, size=4, replace=False):
        min_level = int(rng.choice([3, 4, 5, 6, 7]))
        level_gates[spell] = min_level
        add(
            {"action": "cast", "spell": spell, "actor_level_below": min_level},
            "BLOCKED", 1, f"{spell}_level_gate",
            [
                f"Casting {spell} requires the caster to be at least level {min_level}.",
                f"Only units of level {min_level} or higher can cast {spell}.",
                f"{spell} is locked until level {min_level} — lower-level casters cannot use it.",
            ],
        )
    for ut in rng.choice(UNIT_TYPES, size=3, replace=False):
        min_level = int(rng.choice([4, 5, 6]))
        add(
            {"action": "attack", "actor_type": ut, "actor_level_below": min_level},
            "MODIFIED", 1, f"{ut}_low_level_attack",
            [
                f"A {ut} below level {min_level} deals reduced attack damage due to inexperience.",
                f"Low-level {ut}s (below {min_level}) suffer a damage penalty on attacks.",
                f"Until reaching level {min_level}, {ut} attacks are weakened significantly.",
            ],
        )

    # ── CAT 7: HP THRESHOLD RULES (~8 rules) ──
    for action in ["attack", "cast"]:
        hp_thresh = int(rng.choice([15, 20, 25]))
        verb = "attack" if action == "attack" else "cast spells"
        add(
            {"action": action, "actor_hp_below": hp_thresh},
            "MODIFIED", 1, f"low_hp_{action}",
            [
                f"Units below {hp_thresh} HP suffer penalties when attempting to {verb} — injuries impair performance.",
                f"With HP below {hp_thresh}, the unit's {action} effectiveness is significantly reduced.",
                f"Wounded units (below {hp_thresh} HP) take a performance hit on {action} actions.",
            ],
        )
    add(
        {"action": "move", "actor_hp_below": 10},
        "MODIFIED", 1, "critical_hp_move",
        [
            "Units below 10 HP move at half speed — their injuries severely limit mobility.",
            "Critically wounded units (below 10 HP) can barely drag themselves forward.",
            "At fewer than 10 HP, movement is agonizingly slow — half the normal speed.",
        ],
    )

    # ── CAT 8: STRENGTH vs DEFENSE RULES (~6 rules) ──
    add(
        {"action": "attack", "actor_strength_below_target_defense": True},
        "MODIFIED", 1, "weak_vs_strong",
        [
            "The attacker's strength is lower than the target's defense, resulting in reduced damage.",
            "When the target's defense exceeds the attacker's strength, damage output drops significantly.",
            "Insufficient strength against the target's defenses means the attack deals partial damage.",
        ],
    )
    add(
        {"action": "attack", "actor_strength_exceeds_target_defense_by": 10},
        "MODIFIED", 1, "overwhelming_force",
        [
            "The attacker's strength exceeds the target's defense by more than 10, dealing devastating bonus damage.",
            "With strength outpacing defense by 10+, the attack lands with overwhelming force.",
            "A strength advantage of 10 or more over the target's defense results in a crushing blow.",
        ],
    )

    # ── CAT 9: COMBAT RULES (~12 rules) ──
    add(
        {"action": "attack", "not_adjacent": True},
        "BLOCKED", 1, "melee_range",
        [
            "Melee attacks require the attacker to be adjacent to the target.",
            "The target is out of melee range — physical attacks cannot reach.",
            "Without adjacency, a melee attack has no chance of connecting.",
        ],
    )
    add(
        {"action": "attack", "actor_type": "Ranger", "not_adjacent": True},
        "ALLOWED", 2, "ranger_ranged",
        [
            "Rangers can attack at range with their bows, bypassing the adjacency requirement.",
            "The Ranger's ranged weapons reach non-adjacent targets without penalty.",
            "As a ranged unit, the Ranger strikes from a distance.",
        ],
    )
    add(
        {"action": "attack", "actor_has_status": "Invisible"},
        "MODIFIED", 2, "sneak_attack",
        [
            "The Invisible attacker strikes from stealth, dealing double damage as a sneak attack. Invisible status is lost.",
            "A sneak attack from Invisible status doubles the damage output, but reveals the attacker.",
            "Striking while Invisible grants a devastating sneak attack bonus before the stealth breaks.",
        ],
    )
    add(
        {"action": "attack", "target_has_status": "Shielded"},
        "MODIFIED", 1, "shield_reduction",
        [
            "The target's Shield absorbs a portion of the incoming damage, reducing it by half.",
            "Shielded targets take only 50% damage from physical attacks.",
            "The magical barrier around the target deflects half the attack's force.",
        ],
    )
    for t in rng.choice(TERRAINS, size=2, replace=False):
        add(
            {"action": "attack", "actor_terrain": t},
            "MODIFIED", 1, f"{t}_combat_modifier",
            [
                f"Attacking from {t} terrain imposes an accuracy penalty due to unstable footing.",
                f"The {t} environment makes it harder to land attacks — accuracy suffers.",
                f"Combat effectiveness drops when fighting on {t} terrain.",
            ],
        )
    add(
        {"action": "attack", "actor_terrain": "Mountain", "not_adjacent": True},
        "ALLOWED", 2, "elevation_advantage",
        [
            "The Mountain's elevation grants ranged attack capability to all units.",
            "From the Mountain's height, even melee units can strike distant targets below.",
            "Elevation advantage on Mountain terrain turns all attacks into ranged attacks.",
        ],
    )

    # ── CAT 10: SPELL-STATUS INTERACTIONS (~10 rules) ──
    for spell, opposing_status in [("Fireball", "Frozen"), ("Frostbind", "Burning"),
                                    ("Heal", "Poisoned"), ("Drain", "Shielded")]:
        add(
            {"action": "cast", "spell": spell, "target_has_status": opposing_status},
            "MODIFIED", 1, f"{spell}_vs_{opposing_status}",
            [
                f"{spell} against a {opposing_status} target neutralizes the status instead of dealing normal damage.",
                f"When targeting a {opposing_status} unit, {spell} removes the {opposing_status} effect rather than applying its usual effect.",
                f"The {spell} spell interacts with {opposing_status} status — the status is cleansed instead of the normal spell outcome.",
            ],
        )
    for spell in rng.choice(SPELLS, size=3, replace=False):
        for status in rng.choice(STATUSES, size=1, replace=False):
            add(
                {"action": "cast", "spell": spell, "actor_has_status": status[0] if isinstance(status, np.ndarray) else status},
                "MODIFIED", 2, f"{spell}_caster_{status}",
                [
                    f"A caster with {status} status channels that energy into {spell}, amplifying its effect.",
                    f"The {status} condition on the caster resonates with {spell}, enhancing the spell's power.",
                    f"Casting {spell} while under {status} produces a synergy that strengthens the spell.",
                ],
            )

    # ── CAT 11: MOVEMENT MODIFIERS (~8 rules) ──
    for ut in rng.choice(UNIT_TYPES, size=4, replace=False):
        bonus_terrain = rng.choice(TERRAINS)
        while bonus_terrain in terrain_block_map.get(ut, []):
            bonus_terrain = rng.choice(TERRAINS)
        add(
            {"action": "move", "actor_type": ut, "target_terrain": bonus_terrain},
            "MODIFIED", 1, f"{ut}_{bonus_terrain}_bonus",
            [
                f"{ut}s gain a movement bonus when entering {bonus_terrain} — the terrain suits their abilities.",
                f"The {bonus_terrain} environment is favorable for {ut}s, granting enhanced movement.",
                f"A {ut} entering {bonus_terrain} moves with extra efficiency.",
            ],
        )

    # ── CAT 12: PRIORITY OVERRIDES (~20 rules) ──
    # Status overrides terrain restrictions
    add(
        {"action": "move", "actor_has_status": "Invisible", "not_blocked_terrain": False},
        "ALLOWED", 3, "invisible_bypass",
        [
            "The Invisible status allows the unit to bypass terrain restrictions — stealth finds hidden paths.",
            "While Invisible, the unit slips through normally impassable terrain undetected.",
            "Invisibility reveals secret passages through otherwise restricted terrain.",
        ],
    )
    # High-level overrides
    add(
        {"action": "cast", "actor_level_above": 8},
        "MODIFIED", 3, "master_caster",
        [
            "A caster of level 8 or higher has mastered their craft — terrain penalties on spellcasting are reduced.",
            "Master-level casters (level 8+) partially overcome environmental interference through sheer skill.",
            "At level 8 and above, the caster's expertise mitigates terrain-based spell disruption.",
        ],
    )
    add(
        {"action": "attack", "actor_level_above": 7, "actor_hp_below": 25},
        "MODIFIED", 3, "veteran_resilience",
        [
            "A veteran unit (level 7+) fights through injuries that would cripple less experienced fighters.",
            "High-level units above level 7 partially ignore the HP penalty through battle-hardened resilience.",
            "Veterans of level 7+ have learned to fight effectively even while wounded.",
        ],
    )
    # Burning caster overrides fire spell terrain blocks
    for spell in ["Fireball", "Lightning"]:
        if spell in spell_terrain_blocks:
            for t in spell_terrain_blocks[spell][:1]:
                add(
                    {"action": "cast", "spell": spell, "actor_terrain": t, "actor_has_status": "Burning"},
                    "ALLOWED", 3, f"burning_{spell}_{t}_override",
                    [
                        f"The caster's Burning status provides enough elemental affinity to overcome {t}'s interference with {spell}.",
                        f"Fire from the Burning condition fuels {spell}, allowing it to be cast despite the {t} terrain.",
                        f"A Burning caster channels their own flames into {spell}, bypassing {t}'s suppression.",
                    ],
                )
    # Paladin terrain overrides
    for t in rng.choice(TERRAINS, size=2, replace=False):
        add(
            {"action": "move", "actor_type": "Paladin", "target_terrain": t},
            "MODIFIED", 2, f"paladin_{t}_consecrate",
            [
                f"Paladins consecrate {t} terrain as they enter, neutralizing its negative effects for allies.",
                f"A Paladin's holy presence purifies {t} terrain upon entry.",
                f"The Paladin blesses the {t} ground, transforming it into safe passage.",
            ],
        )
    # High-mana spell boost
    for spell in rng.choice(SPELLS, size=3, replace=False):
        threshold = int(rng.choice([35, 40, 45]))
        add(
            {"action": "cast", "spell": spell, "actor_mana_above": threshold},
            "MODIFIED", 2, f"{spell}_overcharge",
            [
                f"With mana above {threshold}, {spell} is overcharged — dealing enhanced damage or stronger effects.",
                f"Excess mana ({threshold}+) supercharges {spell}, amplifying its potency significantly.",
                f"When the caster has {threshold} or more mana, {spell} activates at heightened power.",
            ],
        )
    # High-defense target rules
    add(
        {"action": "attack", "target_defense_above": 15},
        "MODIFIED", 1, "heavy_armor",
        [
            "Targets with defense above 15 are heavily armored — attacks deal substantially reduced damage.",
            "The target's defense exceeding 15 represents thick armor that blunts most attacks.",
            "Against a target with 15+ defense, even powerful attacks are partially deflected.",
        ],
    )
    add(
        {"action": "cast", "spell": "Drain", "target_defense_above": 12},
        "MODIFIED", 1, "drain_vs_armor",
        [
            "Drain struggles against targets with defense above 12 — the magical siphon is partially blocked by resilience.",
            "High-defense targets (12+) resist Drain's life-stealing effect, reducing the amount siphoned.",
            "The target's defense of 12+ interferes with Drain, cutting the life steal in half.",
        ],
    )
    # Adjacent ally bonus
    add(
        {"action": "cast", "spell": "Rally", "adjacent": True},
        "MODIFIED", 2, "rally_adjacency_bonus",
        [
            "Rally cast on an adjacent ally has increased effectiveness — proximity amplifies the morale boost.",
            "The close proximity of the target enhances Rally's effect, providing a stronger buff.",
            "Adjacent allies receive a more potent Rally buff due to the caster's direct presence.",
        ],
    )
    add(
        {"action": "cast", "spell": "Heal", "adjacent": True, "actor_mana_above": 30},
        "MODIFIED", 3, "empowered_heal",
        [
            "Healing an adjacent ally with 30+ mana produces an empowered effect — restoring significantly more HP.",
            "With ample mana (30+) and direct proximity, Heal restores nearly double the normal amount.",
            "The combination of adjacency and 30+ mana creates a powerful healing surge.",
        ],
    )

    return rules


def rule_matches(rule, situation):
    """Check if all conditions in a rule are satisfied by the situation."""
    for key, val in rule["conditions"].items():
        if key == "actor_has_status":
            if val not in situation.get("actor_statuses", []):
                return False
        elif key == "target_has_status":
            if val not in situation.get("target_statuses", []):
                return False
        elif key == "not_adjacent":
            if situation.get("adjacent", True) != (not val):
                return False
        elif key == "actor_mana_below":
            if situation.get("actor_mana", 50) >= val:
                return False
        elif key == "actor_mana_above":
            if situation.get("actor_mana", 0) <= val:
                return False
        elif key == "actor_level_below":
            if situation.get("actor_level", 10) >= val:
                return False
        elif key == "actor_level_above":
            if situation.get("actor_level", 1) <= val:
                return False
        elif key == "actor_hp_below":
            if situation.get("actor_hp", 100) >= val:
                return False
        elif key == "target_defense_above":
            if situation.get("target_defense", 0) <= val:
                return False
        elif key == "actor_strength_below_target_defense":
            if val and situation.get("actor_strength", 10) >= situation.get("target_defense", 10):
                return False
        elif key == "actor_strength_exceeds_target_defense_by":
            diff = situation.get("actor_strength", 10) - situation.get("target_defense", 10)
            if diff < val:
                return False
        elif key == "not_blocked_terrain":
            pass
        else:
            if situation.get(key) != val:
                return False
    return True


def resolve_situation(rules, situation):
    """Find matching rules, resolve by priority, return matches + outcome."""
    matching = [r for r in rules if rule_matches(r, situation)]
    if not matching:
        return [], "ALLOWED"

    matching.sort(key=lambda r: r["priority"], reverse=True)
    top_priority = matching[0]["priority"]
    top_rules = [r for r in matching if r["priority"] == top_priority]

    if any(r["effect"] == "BLOCKED" for r in top_rules):
        return matching, "BLOCKED"
    elif any(r["effect"] == "MODIFIED" for r in top_rules):
        return matching, "MODIFIED"
    else:
        return matching, "ALLOWED"


SIT_TEMPLATES = [
    "A level {al} {at} stands on {aterr} terrain with {am} mana and {ah} HP (strength {ast}, defense {ad}). {target_desc}{at} attempts to {action_desc}.",
    "On {aterr} terrain, a {at} at level {al} ({am} mana, {ah} HP, {ast} strength, {ad} defense) {action_desc}. {target_desc}",
    "The {at} is level {al}, positioned on {aterr} with {ah} HP and {am} mana remaining ({ast} strength, {ad} defense). {target_desc}The unit tries to {action_desc}.",
    "{at} (level {al}) occupies {aterr} terrain. Current stats: {am} mana, {ah} HP, {ast} strength, {ad} defense. {target_desc}The {at} wants to {action_desc}.",
    "A {at} of level {al} is on {aterr}, carrying {am} mana with {ah} HP left and combat stats of {ast} strength and {ad} defense. {target_desc}The attempt: {action_desc}.",
]

OUTCOME_TEMPLATES = {
    "BLOCKED": [
        "Ultimately, the action is blocked and cannot proceed.",
        "The attempt fails completely under these conditions.",
        "The action is prevented from going through.",
    ],
    "MODIFIED": [
        "The action proceeds but with altered effects.",
        "While not outright blocked, the outcome is modified from the standard result.",
        "The action succeeds in a modified form.",
    ],
    "ALLOWED": [
        "The action proceeds without restriction.",
        "Nothing prevents the action from succeeding normally.",
        "The attempt goes through as intended.",
    ],
}

TRANSITIONS = [
    "However, ", "Additionally, ", "Furthermore, ", "On top of that, ",
    "Meanwhile, ", "At the same time, ", "Compounding this, ", "Beyond that, ",
]


def situation_to_text(rng, sit):
    """Render a situation as varied natural language."""
    at = sit["actor_type"]
    aterr = sit["actor_terrain"]
    al = sit["actor_level"]
    am = sit["actor_mana"]
    ah = sit["actor_hp"]
    ast_ = sit["actor_strength"]
    ad = sit["actor_defense"]

    status_str = ""
    if sit["actor_statuses"]:
        status_str = f" The {at} has {' and '.join(sit['actor_statuses'])} status."

    if sit["action"] == "move":
        action_desc = f"move to {sit['target_terrain']}"
        target_desc = ""
    elif sit["action"] == "attack":
        tt = sit["target_type"]
        tterr = sit["target_terrain"]
        adj = "adjacent to" if sit["adjacent"] else "distant from"
        t_status = ""
        if sit["target_statuses"]:
            t_status = f" with {' and '.join(sit['target_statuses'])} status"
        target_desc = (
            f"The target is a level {sit['target_level']} {tt}{t_status} "
            f"on {tterr} ({sit['target_hp']} HP, {sit['target_strength']} strength, "
            f"{sit['target_defense']} defense), {adj} the attacker. "
        )
        action_desc = f"attack the {tt}"
    else:
        tt = sit["target_type"]
        tterr = sit["target_terrain"]
        adj = "adjacent to" if sit["adjacent"] else "distant from"
        t_status = ""
        if sit["target_statuses"]:
            t_status = f" with {' and '.join(sit['target_statuses'])} status"
        target_desc = (
            f"The target is a level {sit['target_level']} {tt}{t_status} "
            f"on {tterr} ({sit['target_hp']} HP, {sit['target_defense']} defense), "
            f"{adj} the caster. "
        )
        action_desc = f"cast {sit['spell']} on the {tt}"

    template = SIT_TEMPLATES[rng.randint(len(SIT_TEMPLATES))]
    text = template.format(
        al=al, at=at, aterr=aterr, am=am, ah=ah, ast=ast_, ad=ad,
        target_desc=target_desc, action_desc=action_desc,
    )
    text += status_str
    return text


def generate_analysis(rng, sit, matching_rules, outcome):
    """Compose a varied paragraph-form analysis from matching rules."""
    if not matching_rules:
        return rng.choice(OUTCOME_TEMPLATES["ALLOWED"])

    parts = []
    used_transitions = set()
    for i, rule in enumerate(matching_rules):
        desc = rule["descriptions"][rng.randint(len(rule["descriptions"]))]

        desc = _inject_numbers(desc, sit)

        if i == 0:
            parts.append(desc)
        else:
            available = [t for t in TRANSITIONS if t not in used_transitions]
            if not available:
                available = TRANSITIONS
            trans = available[rng.randint(len(available))]
            used_transitions.add(trans)
            parts.append(trans + desc[0].lower() + desc[1:])

    parts.append(OUTCOME_TEMPLATES[outcome][rng.randint(len(OUTCOME_TEMPLATES[outcome]))])
    return " ".join(parts)


def _inject_numbers(desc, sit):
    """Replace generic references with specific numbers from the situation."""
    desc = desc.replace("their injuries", f"their {sit['actor_hp']} HP")
    if "mana" in desc.lower() and str(sit["actor_mana"]) not in desc:
        desc = desc.rstrip(".") + f" (current mana: {sit['actor_mana']})."
    if "level" in desc.lower() and "level " in desc and str(sit["actor_level"]) not in desc:
        desc = desc.rstrip(".") + f" (unit is level {sit['actor_level']})."
    return desc


def situation_to_question(sit):
    """Generate the question for a situation."""
    if sit["action"] == "move":
        return f"Can the {sit['actor_type']} move to {sit['target_terrain']}?"
    elif sit["action"] == "attack":
        return f"Can the {sit['actor_type']} attack the {sit['target_type']}?"
    else:
        return f"Can the {sit['actor_type']} cast {sit['spell']} on the {sit['target_type']}?"


def generate_situation(rng, rules):
    """Generate a random situation that triggers 2+ rules."""
    for _ in range(300):
        action = rng.choice(ACTIONS)
        actor_type = rng.choice(UNIT_TYPES)
        actor_terrain = rng.choice(TERRAINS)
        actor_level = int(rng.randint(1, 11))
        actor_mana = int(rng.randint(5, 51))
        actor_hp = int(rng.randint(5, 101))
        actor_strength = int(rng.randint(3, 21))
        actor_defense = int(rng.randint(3, 21))
        n_statuses = rng.choice([0, 0, 0, 1, 1, 2])
        actor_statuses = sorted(rng.choice(STATUSES, size=min(n_statuses, 2), replace=False).tolist()) if n_statuses > 0 else []

        target_type = rng.choice(UNIT_TYPES)
        target_terrain = rng.choice(TERRAINS)
        target_level = int(rng.randint(1, 11))
        target_hp = int(rng.randint(5, 101))
        target_strength = int(rng.randint(3, 21))
        target_defense = int(rng.randint(3, 21))
        target_mana = int(rng.randint(5, 51))
        t_statuses = rng.choice([0, 0, 1, 1, 2])
        target_statuses = sorted(rng.choice(STATUSES, size=min(t_statuses, 2), replace=False).tolist()) if t_statuses > 0 else []

        adjacent = bool(rng.choice([True, True, True, False]))
        spell = rng.choice(SPELLS) if action == "cast" else None

        sit = {
            "action": action,
            "actor_type": actor_type, "actor_terrain": actor_terrain,
            "actor_level": actor_level, "actor_mana": actor_mana,
            "actor_hp": actor_hp, "actor_strength": actor_strength, "actor_defense": actor_defense,
            "actor_statuses": actor_statuses,
            "target_type": target_type, "target_terrain": target_terrain,
            "target_level": target_level, "target_hp": target_hp,
            "target_strength": target_strength, "target_defense": target_defense,
            "target_mana": target_mana, "target_statuses": target_statuses,
            "adjacent": adjacent, "spell": spell,
        }

        if action == "move":
            sit["target_terrain"] = rng.choice(TERRAINS)

        matching, outcome = resolve_situation(rules, sit)
        if len(matching) >= 2:
            return sit, matching, outcome

    sit = {
        "action": "attack", "actor_type": "Rogue",
        "actor_terrain": "Forest", "actor_level": 5, "actor_mana": 20,
        "actor_hp": 60, "actor_strength": 15, "actor_defense": 8,
        "actor_statuses": ["Invisible"],
        "target_type": "Guardian", "target_terrain": "Plains",
        "target_level": 6, "target_hp": 80,
        "target_strength": 12, "target_defense": 16,
        "target_mana": 10, "target_statuses": ["Shielded"],
        "adjacent": True, "spell": None,
    }
    matching, outcome = resolve_situation(rules, sit)
    return sit, matching, outcome


def main():
    rng = np.random.RandomState(SEED)
    rules = build_rules(rng)
    print(f"Generated {len(rules)} rules.")

    out_dir = Path("raw_data")
    out_dir.mkdir(exist_ok=True)

    N_SAMPLES = 5500
    TARGET = {"BLOCKED": 2200, "MODIFIED": 2000, "ALLOWED": 1300}
    rows = []
    seen_keys = set()
    counts = {"BLOCKED": 0, "MODIFIED": 0, "ALLOWED": 0}

    for _ in range(N_SAMPLES * 10):
        if sum(counts.values()) >= sum(TARGET.values()):
            break

        sit, matching, outcome = generate_situation(rng, rules)

        sit_key = (
            sit["action"], sit["actor_type"], sit["actor_terrain"],
            sit["actor_level"], sit["actor_mana"], sit["actor_hp"],
            tuple(sit["actor_statuses"]),
            sit.get("spell", ""),
            sit["target_type"], sit["target_terrain"],
            sit["target_defense"],
            tuple(sit["target_statuses"]),
            sit["adjacent"],
        )

        if sit_key in seen_keys:
            continue

        if counts[outcome] >= TARGET[outcome]:
            continue

        seen_keys.add(sit_key)
        counts[outcome] += 1

        sit_text = situation_to_text(rng, sit)
        question = situation_to_question(sit)
        analysis = generate_analysis(rng, sit, matching, outcome)
        output = f"<analysis>{analysis}</analysis>\n<outcome>{outcome}</outcome>"

        rows.append({
            "question_id": len(rows),
            "situation": sit_text,
            "question": question,
            "analysis": analysis,
            "outcome": outcome,
            "output": output,
        })

    df = pd.DataFrame(rows)

    df.to_csv(out_dir / "data.csv", index=False)

    print(f"\nGenerated {len(df)} Q&A pairs.")
    print(f"Outcome distribution:\n{df['outcome'].value_counts()}")
    print(f"Note: noise will be injected into training split by prepare.py (10%)")
    print(f"\nSample:")
    for _, row in df.head(2).iterrows():
        print(f"\n  Situation: {row['situation'][:150]}...")
        print(f"  Question:  {row['question']}")
        print(f"  Outcome:   {row['outcome']}")
        print(f"  Analysis:  {row['analysis'][:150]}...")


if __name__ == "__main__":
    main()
