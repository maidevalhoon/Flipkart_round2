"""
Tests for recommend.py — face-validity assertions on rule logic.
No ML needed: recommendation is purely rule-based.
Run with: pytest tests/test_recommend.py -v
"""
import os, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.recommend import recommend, fallback_recommend
from src.config import RULES


def make_input(**kwargs):
    base = {
        "severity": "Low",
        "probability": 0.3,
        "event_cause": "vehicle_breakdown",
        "requires_road_closure": False,
        "is_corridor": 0,
        "event_type": "unplanned",
        "hour": 14,
        "hour_bucket": "midday",
        "dup_cluster_size": 1,
        "corridor_name": "Non-corridor",
    }
    base.update(kwargs)
    return base


# ─────────────────────────────────────────────────────────────────────────────
def test_high_base_manpower_greater_than_low():
    high = recommend(make_input(severity="High"))
    low  = recommend(make_input(severity="Low"))
    assert high["manpower_count"] > low["manpower_count"]


def test_road_closure_adds_manpower():
    without = recommend(make_input(severity="High", requires_road_closure=False))
    with_   = recommend(make_input(severity="High", requires_road_closure=True))
    assert with_["manpower_count"] >= without["manpower_count"] + RULES["closure_bonus_manpower"]


def test_crowd_event_adds_manpower():
    normal = recommend(make_input(severity="High", event_cause="vehicle_breakdown"))
    crowd  = recommend(make_input(severity="High", event_cause="procession"))
    assert crowd["manpower_count"] >= normal["manpower_count"] + RULES["crowd_bonus_manpower"]


def test_closure_true_adds_barricades():
    r = recommend(make_input(severity="High", requires_road_closure=True))
    assert r["barricade_count"] >= RULES["base_barricades"]["closure"]


def test_low_no_closure_zero_barricades():
    r = recommend(make_input(severity="Low", requires_road_closure=False))
    assert r["barricade_count"] == RULES["base_barricades"]["Low_no_closure"]


def test_diversion_suggested_on_closure():
    r = recommend(make_input(severity="High", requires_road_closure=True))
    assert r["diversion_suggested"] is True


def test_diversion_not_suggested_low_no_closure():
    r = recommend(make_input(severity="Low", requires_road_closure=False,
                              event_cause="vehicle_breakdown", is_corridor=0))
    assert r["diversion_suggested"] is False


def test_rationale_non_empty():
    r = recommend(make_input(severity="High", requires_road_closure=True,
                              event_cause="procession"))
    assert len(r["rationale"]) > 0


def test_hotspot_bonus_capped():
    r_big = recommend(make_input(severity="High", dup_cluster_size=100))
    r_small = recommend(make_input(severity="High", dup_cluster_size=2))
    bonus_big = r_big["manpower_count"] - RULES["base_manpower"]["High"]
    bonus_small = r_small["manpower_count"] - RULES["base_manpower"]["High"]
    # Hotspot bonus for big cluster should not exceed cap
    assert bonus_big - bonus_small <= RULES["hotspot_cap"]


def test_fallback_returns_valid_output():
    r = fallback_recommend("vehicle_breakdown", requires_road_closure=True)
    assert "manpower_count" in r
    assert "severity" in r
    assert r["manpower_count"] >= 1


def test_fallback_crowd_higher_manpower():
    r_vb = fallback_recommend("vehicle_breakdown", requires_road_closure=False)
    r_pe = fallback_recommend("procession", requires_road_closure=False)
    assert r_pe["manpower_count"] >= r_vb["manpower_count"]


def test_peak_hour_bonus_only_when_high():
    peak_high = recommend(make_input(severity="High", hour=20, hour_bucket="late"))
    peak_low  = recommend(make_input(severity="Low", hour=20, hour_bucket="late"))
    off_high  = recommend(make_input(severity="High", hour=14, hour_bucket="midday"))
    # High peak > High off-peak
    assert peak_high["manpower_count"] >= off_high["manpower_count"]
    # Low peak should NOT get the bonus
    assert peak_low["manpower_count"] < peak_high["manpower_count"]
