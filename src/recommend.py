"""
recommend.py — Rule-based recommendation engine.

Maps predicted severity + event attributes → manpower/barricade/diversion.
All thresholds and bonuses live in config.RULES — tune there, not here.

No ML: there is no deployment-outcome ground truth in the dataset.
Fallback fires when model is unavailable or probability is in dead-band [0.4, 0.6].
"""
from src.config import RULES

CROWD_CAUSES = {"public_event", "procession", "protest", "vip_movement"}
PEAK_HOURS = {5, 6, 19, 20, 21, 22}


def recommend(event: dict) -> dict:
    """
    Parameters
    ----------
    event : dict with keys:
        severity           : "High" | "Low"
        probability        : float  (model confidence)
        event_cause        : str
        requires_road_closure : bool
        is_corridor        : int (0|1)
        event_type         : "planned" | "unplanned"
        hour               : int  (IST hour 0-23)
        hour_bucket        : str
        dup_cluster_size   : int
        corridor_name      : str

    Returns
    -------
    dict with manpower_count, barricade_count, barricade_placement,
              diversion_suggested, diversion_note, rationale (list[str])
    """
    R = RULES
    severity = event["severity"]
    cause = str(event.get("event_cause", "other")).lower()
    closure = bool(event.get("requires_road_closure", False))
    is_corridor = bool(event.get("is_corridor", 0))
    hour = int(event.get("hour", 12))
    dup = int(event.get("dup_cluster_size", 1))
    corridor_name = str(event.get("corridor_name", ""))
    rationale = []

    # ── Manpower ─────────────────────────────────────────────────────────────
    manpower = R["base_manpower"][severity]
    rationale.append(f"Base manpower ({severity}): {manpower}")

    if closure:
        manpower += R["closure_bonus_manpower"]
        rationale.append(f"+{R['closure_bonus_manpower']} road closure")

    if cause in CROWD_CAUSES:
        manpower += R["crowd_bonus_manpower"]
        rationale.append(f"+{R['crowd_bonus_manpower']} crowd event ({cause})")

    if is_corridor:
        manpower += R["corridor_bonus_manpower"]
        rationale.append(f"+{R['corridor_bonus_manpower']} major corridor")

    if severity == "High" and hour in PEAK_HOURS:
        manpower += R["peak_hour_bonus_manpower"]
        rationale.append(f"+{R['peak_hour_bonus_manpower']} peak hour (IST {hour}:00)")

    hotspot_bonus = min(dup // 2, R["hotspot_cap"])
    if hotspot_bonus > 0:
        manpower += hotspot_bonus
        rationale.append(f"+{hotspot_bonus} repeated hotspot (cluster size={dup})")

    # ── Barricades ───────────────────────────────────────────────────────────
    if closure:
        barricades = R["base_barricades"]["closure"]
        if cause in CROWD_CAUSES:
            barricades += R["crowd_barricade_bonus"]
            rationale.append(f"+{R['crowd_barricade_bonus']} crowd barricade bonus")
    elif severity == "High":
        barricades = R["base_barricades"]["High_no_closure"]
    else:
        barricades = R["base_barricades"]["Low_no_closure"]

    # ── Barricade placement ───────────────────────────────────────────────────
    if barricades > 0 and corridor_name and corridor_name.lower() not in ("", "non-corridor", "unknown"):
        placement = f"Entry/exit points of {corridor_name}"
    elif barricades > 0:
        placement = "Nearest intersection to incident location"
    else:
        placement = "None required"

    # ── Diversion ─────────────────────────────────────────────────────────────
    diversion = closure or (severity == "High" and is_corridor) or (cause in R["diversion_causes"])
    if diversion:
        diversion_note = (
            f"Activate alternate route via parallel corridor to {corridor_name}"
            if corridor_name and corridor_name.lower() not in ("", "non-corridor", "unknown")
            else "Engage traffic personnel to guide flow around incident"
        )
    else:
        diversion_note = "No diversion required"

    return {
        "manpower_count": manpower,
        "barricade_count": barricades,
        "barricade_placement": placement,
        "diversion_suggested": diversion,
        "diversion_note": diversion_note,
        "rationale": rationale,
    }


def fallback_recommend(event_cause: str, requires_road_closure: bool = False) -> dict:
    """
    Used when model is unavailable or confidence is in dead-band [0.4, 0.6].
    Falls back to historical High-rate per cause from EDA.
    """
    R = RULES
    cause = str(event_cause).lower()
    high_rate = R["cause_high_rate"].get(cause, R["cause_high_rate"]["default"])
    severity = "High" if high_rate >= R["fallback_high_threshold"] else "Low"

    result = recommend({
        "severity": severity,
        "probability": high_rate,
        "event_cause": cause,
        "requires_road_closure": requires_road_closure,
        "is_corridor": 0,
        "event_type": "unplanned",
        "hour": 12,
        "hour_bucket": "midday",
        "dup_cluster_size": 1,
        "corridor_name": "",
    })
    result["severity"] = severity
    result["fallback_used"] = True
    result["rationale"].insert(0, f"FALLBACK: model unavailable; using cause_high_rate={high_rate:.2f}")
    return result
