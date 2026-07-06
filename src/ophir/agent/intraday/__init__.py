"""Intraday research pipeline (Option B) — dollar bars + triple-barrier labels.

An **additive** subpackage that builds a genuinely intraday forecasting model to
drop into the ophir signal slot. See ``ophir-bot/intraday-model-blueprint.md`` for
the design and ``.claude/plans`` for the free-tier-data test plan. Nothing here is
imported by the daily ``ophir trade`` path, which is left untouched.

Phases: P0 data (``data``) -> P1 features/labels -> P2 baseline + cost gate.
"""
