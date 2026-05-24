from lci_reduce.errors import UncharacterisedExchangeError
from lci_reduce.models import FlowInfo, ImpactCategory
from lci_reduce.cli import build_parser
from lci_reduce.reducer import reduce_process


def make_process():
    return {
        "@id": "process-1",
        "name": "Toy Process",
        "exchanges": [
            {
                "@id": "e1",
                "amount": 10.0,
                "flow": {"@id": "flow-1", "name": "CO2"},
                "unit": {"name": "kg"},
            }
        ],
    }


def make_flows():
    return {
        "flow-1": FlowInfo(
            flow_id="flow-1",
            name="CO2",
            flow_type="ELEMENTARY_FLOW",
            category_path="air/urban air",
            is_elementary=True,
            raw={},
        )
    }


def make_categories():
    return [
        ImpactCategory(
            category_id="cat-1",
            name="Climate change",
            method_id="m1",
            method_name="IPCC",
            path="",
            metadata_text="",
            reference_unit="kg",
            factors={},
            raw={},
        )
    ]


def test_cli_default_uncharacterised_policy_is_drop():
    args = build_parser().parse_args(
        [
            "create",
            "--database",
            "database.zip",
            "--output",
            "out",
            "--tau",
            "0.95",
            "--method-selection",
            "all",
        ]
    )
    assert args.uncharacterised_policy == "drop"


def test_uncharacterised_keep():
    result = reduce_process(
        process_data=make_process(),
        flow_lookup=make_flows(),
        categories=make_categories(),
        tau=0.95,
        uncharacterised_policy="keep",
        strict_units=True,
        tol=1e-12,
        database_name="db",
    )
    assert result.process_row["n_elementary_after"] == 1
    assert result.elementary_rows[0]["protected"] is True


def test_uncharacterised_drop():
    result = reduce_process(
        process_data=make_process(),
        flow_lookup=make_flows(),
        categories=make_categories(),
        tau=0.95,
        uncharacterised_policy="drop",
        strict_units=True,
        tol=1e-12,
        database_name="db",
    )
    assert result.process_row["n_elementary_after"] == 0
    assert result.elementary_rows[0]["removed"] is True
    assert result.elementary_rows[0]["removal_reason"] == "uncharacterised_drop"
    assert result.process_row["n_uncharacterised_kept"] == 0
    assert result.process_row["n_uncharacterised_removed"] == 1


def test_uncharacterised_fail():
    try:
        reduce_process(
            process_data=make_process(),
            flow_lookup=make_flows(),
            categories=make_categories(),
            tau=0.95,
            uncharacterised_policy="fail",
            strict_units=True,
            tol=1e-12,
            database_name="db",
        )
    except UncharacterisedExchangeError:
        return
    raise AssertionError("Expected UncharacterisedExchangeError")
