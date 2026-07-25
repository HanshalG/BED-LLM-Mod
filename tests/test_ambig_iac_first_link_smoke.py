import json

import pytest

from scripts.ambig_iac_first_link_smoke import (
    Feature,
    _parse_population,
    best_followup,
    feature_entropy,
    fixed_depth_two_score,
    medoid_index,
    parse_generated_spec,
    realized_endpoint,
    regenerated_depth_two_score,
    select_root_features,
    spec_features,
    spec_similarity,
    target_spec_from_plan,
)


def _spec(resources, topology=None, attributes=None):
    return {
        "resources": resources,
        "topology": topology or {},
        "attributes": attributes or {},
    }


def test_generated_spec_parser_and_features():
    parsed = parse_generated_spec(
        json.dumps(
            _spec(
                {
                    "vpc": "aws_vpc.main",
                    "subnet": "aws_subnet.public",
                    "subnet2": "aws_subnet.private",
                },
                {"subnet": ["vpc"], "subnet2": ["vpc"]},
                {"vpc": {"cidr_block": "10.0.0.0/16"}},
            )
        )
    )
    features = spec_features(parsed)
    assert Feature("resource", "aws_subnet", "2") in features
    assert Feature("topology", "aws_subnet", "aws_vpc") in features
    assert Feature("attribute", "aws_vpc", "cidr_block") in features

    with pytest.raises(ValueError, match="unknown resource"):
        parse_generated_spec(
            json.dumps(
                _spec(
                    {"vpc": "aws_vpc.main"},
                    {"missing": ["vpc"]},
                )
            )
        )


def test_entropy_and_balanced_root_selection():
    resource = Feature("resource", "aws_vpc", "1")
    topology = Feature("topology", "aws_subnet", "aws_vpc")
    attribute = Feature("attribute", "aws_vpc", "cidr_block")
    extra = Feature("resource", "aws_s3_bucket", "1")
    particles = [
        frozenset({resource, topology, attribute}),
        frozenset({resource, topology}),
        frozenset({extra, attribute}),
        frozenset({extra}),
    ]
    assert feature_entropy(particles, resource) == pytest.approx(
        0.6931471805599453
    )
    roots = select_root_features(particles, count=4)
    assert len(roots) == 4
    assert {root.dimension for root in roots} == {
        "resource",
        "topology",
        "attribute",
    }
    followup, entropy = best_followup(particles, exclude=(resource,))
    assert followup is not None
    assert entropy > 0.0


def test_branch_population_rejects_clarification_contradictions():
    root = Feature("resource", "aws_vpc", "1")
    responses = [
        json.dumps(_spec({"vpc": "aws_vpc.main"})),
        json.dumps(_spec({"vpc": "aws_vpc.secondary"})),
        json.dumps(_spec({"vpc": "aws_vpc.tertiary"})),
        json.dumps(_spec({"vpc": "aws_vpc.fourth"})),
        json.dumps(_spec({"bucket": "aws_s3_bucket.store"})),
    ]
    particles, errors = _parse_population(
        responses,
        history=((root, True),),
    )
    assert len(particles) == 4
    assert len(errors) == 1
    assert "contradicts clarification history" in errors[0]["error"]


def test_target_plan_preserves_reference_attributes_and_dependencies(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "configuration": {
                    "root_module": {
                        "resources": [
                            {
                                "address": "aws_vpc.main",
                                "expressions": {
                                    "cidr_block": {
                                        "constant_value": "10.0.0.0/16"
                                    }
                                },
                            },
                            {
                                "address": "aws_subnet.public",
                                "expressions": {
                                    "vpc_id": {
                                        "references": ["aws_vpc.main.id"]
                                    }
                                },
                            },
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    target = target_spec_from_plan(plan_path)
    assert target["topology"]["aws_subnet.public"] == ["aws_vpc.main"]
    assert target["attributes"]["aws_subnet.public"]["vpc_id"] is True


def test_regenerated_score_and_external_endpoint_are_separate():
    root_a = Feature("resource", "aws_vpc", "1")
    root_b = Feature("resource", "aws_s3_bucket", "1")
    future = Feature("topology", "aws_subnet", "aws_vpc")
    initial = [
        frozenset({root_a}),
        frozenset({root_a}),
        frozenset({root_b}),
        frozenset({root_b}),
    ]
    branches_a = {
        False: [frozenset({future}), frozenset()],
        True: [frozenset({future}), frozenset()],
    }
    branches_b = {
        False: [frozenset({future}), frozenset({future})],
        True: [frozenset({future}), frozenset({future})],
    }
    assert regenerated_depth_two_score(initial, root_a, branches_a) > (
        regenerated_depth_two_score(initial, root_b, branches_b)
    )
    assert fixed_depth_two_score(initial, root_a) == pytest.approx(
        fixed_depth_two_score(initial, root_b)
    )

    target = frozenset({root_a, future})
    endpoint = realized_endpoint(target, root_a, branches_a)
    assert endpoint["root_answer"] is True
    assert endpoint["followup_answer"] is True
    assert endpoint["posterior_particle_count"] == 1
    assert endpoint["selected_similarity"]["combined"] > 0.0


def test_similarity_and_medoid_are_deterministic():
    resource = Feature("resource", "aws_vpc", "1")
    edge = Feature("topology", "aws_subnet", "aws_vpc")
    attribute = Feature("attribute", "aws_vpc", "cidr_block")
    particles = [
        frozenset({resource, edge}),
        frozenset({resource, edge, attribute}),
        frozenset({resource}),
    ]
    assert medoid_index(particles) == 0
    score = spec_similarity(particles[1], particles[0])
    assert score["resource"] == 1.0
    assert score["topology"] == 1.0
    assert score["attribute"] == 0.0
