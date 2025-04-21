from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Callable

from faker import Faker
import numpy as np


class RowFactory:
    """Generate synthetic rows based on simple rule strings.

    **Supported rule syntax** (case‑insensitive):

    * ``fake.<provider>`` ― call the corresponding *Faker* provider, e.g. ``fake.name`` or ``fake.email``
    * ``random_name`` – alias for ``fake.name``  *(convenience)*
    * ``choice(a,b,c)`` ― random element from the provided comma‑separated list
    * ``int_range(min,max)`` ― inclusive uniform integer
    * ``normal(mu,sigma)`` ― float from normal distribution (mean *mu*, stdev *sigma*)
    * ``sequence`` or ``sequence(start)`` ― monotonically increasing integers starting at 1 (or *start*)
    * ``constant(value)`` ― literal string/number
    * **empty / missing rule** ― fallback heuristic based on column name
    """

    _CHOICE_RE = re.compile(r"choice\((.+)\)", re.I)
    _INT_RANGE_RE = re.compile(r"int_range\((\d+),\s*(\d+)\)", re.I)
    _NORMAL_RE = re.compile(r"normal\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)", re.I)
    _CONSTANT_RE = re.compile(r"constant\((.+)\)", re.I)
    _FAKE_RE = re.compile(r"fake\.([A-Za-z0-9_]+)$", re.I)
    _SEQ_RE = re.compile(r"sequence(?:\((\d+)\))?$", re.I)
    _INT_SEQ_RE = re.compile(r"int_sequence\(\s*start\s*=\s*(\d+)\s*\)", re.I)

    _ALIAS_MAP = {
        "random_name": "name",
        "random_first_name": "first_name",
        "random_last_name": "last_name",
        "random_email": "email",
    }
    

    def __init__(self, field_rules: Dict[str, str] | None = None, seed: int | None = None) -> None:
        """Parameters
        ----------
        field_rules
            Mapping ``"Table.Column"`` → rule string (global defaults).
        seed
            Optional RNG seed for reproducibility.  If *None*, a random seed is chosen.
        """
        self.global_rules = field_rules or {}
        if seed is None:
            seed = random.randint(0, 2 ** 32 - 1)
        self.seed = seed

        # PRNG & faker instances
        self.fake = Faker()
        self.fake.seed_instance(seed)
        self.rng = np.random.default_rng(seed)

        # Per‑column sequence counters {fq_col → current_value}
        self._seq_counters: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_rows(
        self,
        *,
        table: str,
        columns: List[str],
        n: int,
        fields_spec: Dict[str, str] | None = None,
        relationships: Dict[str, str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Generate *n* synthetic rows for *table*.

        New signature matches **SyntheticAgent**.
        """
        fields_spec = fields_spec or {}
        relationships = relationships or {}

        # Merge rule sources – per‑call overrides global defaults
        rules: Dict[str, str] = {**self.global_rules, **fields_spec}

        compiled: Dict[str, Callable[[], Any]] = {}
        table_prefix = f"{table}."
        for col in columns:
            fq_col = table_prefix + col
            rule = rules.get(fq_col, "")
            if rule:
                gen_fn = self._compile_rule(rule, fq_col)
            else:
                gen_fn = None  # fallback will be used
            compiled[fq_col] = gen_fn

        rows: List[Dict[str, Any]] = []
        for _ in range(n):
            row: Dict[str, Any] = {}
            for col in columns:
                fq_col = table_prefix + col
                gen_fn = compiled.get(fq_col)
                if gen_fn is None:
                    row[col] = self._fallback_value(col)
                else:
                    row[col] = gen_fn()
            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    # Rule compilation helpers
    # ------------------------------------------------------------------
    def _compile_rule(self, rule: str, fq_col: str) -> Callable[[], Any]:
        """Translate rule text → generator function returning a value."""
        rule = rule.strip()

        # Alias mapping (e.g. random_name → fake.name)
        alias = self._ALIAS_MAP.get(rule.lower())
        if alias:
            return lambda f=self.fake, p=alias: getattr(f, p)()

        # sequence / sequence(start)
        m = self._SEQ_RE.match(rule)
        if m:
            start = int(m.group(1) or 1) - 1  # counter will be incremented before first use
            self._seq_counters[fq_col] = start

            def _next(col=fq_col, ctr=self._seq_counters):
                ctr[col] += 1
                return ctr[col]

            return _next

        # fake.<provider>
        m = self._FAKE_RE.match(rule)
        if m:
            provider = m.group(1)
            if not hasattr(self.fake, provider):
                raise ValueError(f"Unknown Faker provider '{provider}'")
            return lambda f=self.fake, p=provider: getattr(f, p)()

        # choice(a,b,c)
        m = self._CHOICE_RE.match(rule)
        if m:
            raw = m.group(1)
            choices = [c.strip().strip("'\"") for c in raw.split(",")]
            return lambda rng=self.rng, arr=choices: rng.choice(arr)


                # int_sequence(start=42)
        m = self._INT_SEQ_RE.match(rule)
        if m:
            start = int(m.group(1))
            # each field needs its own counter
            # so we wrap in a closure capturing a mutable counter
            counter = {'next': start}
            def seq():
                val = counter['next']
                counter['next'] += 1
                return val
            return seq

        # int_range(min,max)
        m = self._INT_RANGE_RE.match(rule)
        if m:
            lo, hi = map(int, m.groups())
            return lambda rng=self.rng, a=lo, b=hi: int(rng.integers(a, b + 1))

        # normal(mu,sigma)
        m = self._NORMAL_RE.match(rule)
        if m:
            mu, sigma = map(float, m.groups())
            return lambda rng=self.rng, mu=mu, sd=sigma: float(rng.normal(mu, sd))

        # constant(value)
        m = self._CONSTANT_RE.match(rule)
        if m:
            val = m.group(1).strip().strip("'\"")
            return lambda v=val: v

        raise ValueError(f"Unsupported rule syntax: {rule}")

    # ------------------------------------------------------------------
    # Fallback heuristics (very light) — can be expanded per schema needs
    # ------------------------------------------------------------------
    def _fallback_value(self, col: str) -> Any:
        lc = col.lower()
        if lc.endswith("_id") or lc == "id":
            return int(self.rng.integers(1, 1_000_000))
        if "email" in lc:
            return self.fake.email()
        if "date" in lc:
            return self.fake.date()
        if "name" in lc:
            return self.fake.name()
        if "phone" in lc or "mobile" in lc:
            return self.fake.phone_number()
        # default: None (interpreted as NULL in COPY)
        return None


__all__ = ["RowFactory"]
