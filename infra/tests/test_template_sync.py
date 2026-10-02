"""Guards for the CloudFormation template.

The template embeds the real Lambda handler source in its ``ZipFile`` blocks so a
single ``aws cloudformation deploy`` yields working functions. That only holds if
the embedded code stays identical to the tested handler files, so this test fails
whenever they drift, and it also parses the template to catch YAML or intrinsic
errors without needing AWS credentials.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

INFRA = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE = INFRA / "cloudformation" / "airshield-pulse.yaml"

yaml = pytest.importorskip("yaml")


def _load_sync():
    spec = importlib.util.spec_from_file_location(
        "sync_template_code", INFRA / "sync_template_code.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class _CfnLoader(yaml.SafeLoader):
    """Loads CloudFormation short-form intrinsics as plain dicts."""


def _intrinsic(loader, node):  # noqa: ANN001
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node)
    else:
        value = loader.construct_mapping(node)
    return {node.tag: value}


for _tag in (
    "!Ref", "!Sub", "!GetAtt", "!Join", "!Select", "!Split", "!If", "!Equals",
    "!Not", "!And", "!Or", "!FindInMap", "!Base64", "!Cidr", "!ImportValue",
    "!GetAZs", "!Condition",
):
    _CfnLoader.add_constructor(_tag, _intrinsic)


def test_template_zipfile_matches_handler_sources():
    sync = _load_sync()
    assert sync.render() == TEMPLATE.read_text(encoding="utf-8"), (
        "CloudFormation ZipFile blocks drifted from the handler sources; "
        "run: python infra/sync_template_code.py"
    )


def test_template_parses_and_declares_expected_resources():
    template = yaml.load(TEMPLATE.read_text(encoding="utf-8"), Loader=_CfnLoader)
    resources = template["Resources"]

    assert template["AWSTemplateFormatVersion"] == "2010-09-09"

    expected = {
        "DataBucket": "AWS::S3::Bucket",
        "StateTable": "AWS::DynamoDB::Table",
        "SpikeTopic": "AWS::SNS::Topic",
        "IngestFunction": "AWS::Lambda::Function",
        "SpikeFunction": "AWS::Lambda::Function",
        "IngestSchedule": "AWS::Events::Rule",
        "SpikeSchedule": "AWS::Events::Rule",
    }
    for name, kind in expected.items():
        assert name in resources, f"missing resource {name}"
        assert resources[name]["Type"] == kind


def test_embedded_lambda_code_is_importable_and_has_a_handler():
    """The embedded source must actually compile and expose `handler`."""
    sync = _load_sync()
    template = TEMPLATE.read_text(encoding="utf-8")

    matches = list(sync._BLOCK.finditer(template))
    assert len(matches) == 2

    for match, name in zip(matches, ("ingest", "spike")):
        body = match.group("body")
        source = "".join(
            (line[10:] if len(line) > 10 else line[10:]) + "\n" for line in body.split("\n")
        )
        namespace: dict = {}
        exec(compile(source, f"<lambda:{name}>", "exec"), namespace)  # noqa: S102
        assert callable(namespace.get("handler")), f"{name} handler not defined"


def test_no_placeholder_stub_remains_in_template():
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "Placeholder. Deploy the real handler" not in text
    assert "raise RuntimeError(\"deploy the real handler" not in text
