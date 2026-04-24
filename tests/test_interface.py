"""
Tests for the LanguageInterface adapter.

Validates:
  - LLMs produce claims, not beliefs
  - Claims cannot directly update beliefs (enforcement at API level)
  - Formatting helpers return correct epistemic language
"""

import pytest

from episteme.interface import Claim, LanguageInterface
from episteme.models import BeliefType


class TestClaim:
    def test_default_claim_is_theory(self):
        c = Claim(text="The sun rises in the east.")
        assert c.type == BeliefType.THEORY

    def test_claim_confidence_range(self):
        with pytest.raises(ValueError, match="confidence"):
            Claim(text="x", confidence=1.5)

    def test_claim_type_string_coercion(self):
        c = Claim(text="x", type="FACT")
        assert c.type == BeliefType.FACT


class TestLanguageInterface:
    def setup_method(self):
        self.iface = LanguageInterface(llm_name="test-llm", default_domain="science")

    def test_extract_claims_empty_text(self):
        claims = self.iface.extract_claims("")
        assert claims == []

    def test_extract_claims_returns_list(self):
        claims = self.iface.extract_claims("Water boils at 100°C at sea level.")
        assert len(claims) == 1
        assert isinstance(claims[0], Claim)

    def test_extracted_claim_carries_source(self):
        claims = self.iface.extract_claims("Some fact.")
        assert claims[0].source == "test-llm"

    def test_extracted_claim_carries_domain(self):
        claims = self.iface.extract_claims("Some fact.")
        assert claims[0].domain == "science"

    def test_extracted_claim_is_theory(self):
        """LLM outputs are proposals (THEORY) by default."""
        claims = self.iface.extract_claims("Some claim.")
        assert claims[0].type == BeliefType.THEORY

    def test_propose_hypothesis_returns_theory(self):
        claim = self.iface.propose_hypothesis("Maybe dark matter exists.")
        assert claim.type == BeliefType.THEORY
        assert claim.confidence <= 0.5  # hypotheses are low-confidence

    def test_llm_cannot_directly_modify_beliefs(self):
        """
        The LanguageInterface has no method to write to EpistemicMemory.
        Verify the interface exposes no belief-write surface.
        """
        iface = LanguageInterface()
        write_methods = [
            m for m in dir(iface)
            if not m.startswith("_")
            and callable(getattr(iface, m))
            and any(w in m for w in ("add", "revise", "update", "delete", "write", "set"))
        ]
        assert write_methods == [], (
            f"LanguageInterface must not expose write methods: {write_methods}"
        )

    def test_format_belief_active_high_confidence(self):
        text = self.iface.format_belief("Water is H2O.", 0.95, "ACTIVE")
        assert "established" in text.lower() or "likely" in text.lower()

    def test_format_belief_unknown(self):
        text = self.iface.format_belief("X is Y.", 0.5, "UNKNOWN")
        assert "unknown" in text.lower()

    def test_format_belief_disputed(self):
        text = self.iface.format_belief("X is Y.", 0.5, "DISPUTED")
        assert "disputed" in text.lower()

    def test_format_belief_undecided(self):
        text = self.iface.format_belief("X is Y.", 0.5, "UNDECIDED")
        assert "undecided" in text.lower()

    def test_format_unknown_query(self):
        text = self.iface.format_unknown("What is the speed of dark?")
        assert "unknown" in text.lower()

    def test_format_error(self):
        text = self.iface.format_error("revise_belief", "missing evidence")
        assert "error" in text.lower()
        assert "missing evidence" in text

    def test_label_claim_type_default(self):
        btype = self.iface.label_claim_type("The capital of France is Paris.")
        assert isinstance(btype, BeliefType)
