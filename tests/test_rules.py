import os
os.environ["ANTHROPIC_API_KEY"] = "test-key"

from app import RequirementAnalysis, apply_rules

def test_full_high_confidence():
    item = RequirementAnalysis(requirement_id="REQ-001", coverage="full", confidence=0.95, explanation="Clearly supported", evidence=[], skill_gap="")
    result = apply_rules(item)
    assert result.compliance == "YES"
    assert result.score == 90
    assert result.training_required is False

def test_partial_with_gap_requires_training():
    item = RequirementAnalysis(requirement_id="REQ-002", coverage="partial", confidence=0.8, explanation="Partially supported", evidence=[], skill_gap="Kubernetes deployment")
    result = apply_rules(item)
    assert result.compliance == "PARTIAL"
    assert result.score == 60
    assert result.training_required is True
