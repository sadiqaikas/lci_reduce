from lci_reduce.models import CharacterizationFactor, FlowInfo, ImpactCategory
from lci_reduce.reducer import reduce_process


def test_protected_exchange():
    process = {
        "@id": "process-1",
        "name": "Toy Process",
        "exchanges": [
            {
                "@id": "product",
                "amount": 1.0,
                "flow": {"@id": "flow-product", "name": "Product"},
                "unit": {"name": "kg"},
                "quantitativeReference": True,
            },
            {
                "@id": "tech",
                "amount": 2.0,
                "flow": {"@id": "flow-tech", "name": "Input"},
                "unit": {"name": "kg"},
                "provider": {"@id": "provider-1"},
            },
        ],
    }
    flows = {
        "flow-product": FlowInfo("flow-product", "Product", "PRODUCT_FLOW", "", False, {}),
        "flow-tech": FlowInfo("flow-tech", "Input", "PRODUCT_FLOW", "", False, {}),
    }
    category = ImpactCategory("cat", "Climate", "m1", "IPCC", "", "", "kg", {}, {})
    result = reduce_process(
        process_data=process,
        flow_lookup=flows,
        categories=[category],
        tau=0.9,
        uncharacterised_policy="drop",
        strict_units=True,
        tol=1e-12,
        database_name="db",
    )
    assert len(result.reduced_process["exchanges"]) == 2


def test_elementary_provider_link_is_retained_and_reported():
    process = {
        "@id": "process-1",
        "name": "Toy Process",
        "exchanges": [
            {
                "@id": "elem-cover",
                "amount": 1.0,
                "flow": {"@id": "flow-cover", "name": "Cover"},
                "unit": {"name": "kg"},
            },
            {
                "@id": "elem-provider",
                "amount": 1.0,
                "flow": {"@id": "flow-provider", "name": "Protected"},
                "unit": {"name": "kg"},
                "provider": {"@id": "provider-1"},
            },
        ],
    }
    flows = {
        "flow-cover": FlowInfo("flow-cover", "Cover", "ELEMENTARY_FLOW", "air/urban air", True, {}),
        "flow-provider": FlowInfo("flow-provider", "Protected", "ELEMENTARY_FLOW", "air/urban air", True, {}),
    }
    category = ImpactCategory(
        "cat",
        "Climate",
        "m1",
        "IPCC",
        "",
        "",
        "kg",
        {
            "flow-cover": CharacterizationFactor("flow-cover", 0.95, "kg", {}),
            "flow-provider": CharacterizationFactor("flow-provider", 0.05, "kg", {}),
        },
        {},
    )
    result = reduce_process(
        process_data=process,
        flow_lookup=flows,
        categories=[category],
        tau=0.95,
        uncharacterised_policy="drop",
        strict_units=True,
        tol=1e-12,
        database_name="db",
    )
    exchange_ids = [exchange["@id"] for exchange in result.reduced_process["exchanges"]]
    assert exchange_ids == ["elem-cover", "elem-provider"]
    provider_row = next(row for row in result.elementary_rows if row["exchange_id"] == "elem-provider")
    assert provider_row["protected"] is True
    assert provider_row["removal_reason"] == "provider_link"
    assert provider_row["selected"] is True


def test_final_coverage_uses_post_protection_selected_mask():
    process = {
        "@id": "process-1",
        "name": "Toy Process",
        "exchanges": [
            {
                "@id": "elem-cover",
                "amount": 1.0,
                "flow": {"@id": "flow-cover", "name": "Cover"},
                "unit": {"name": "kg"},
            },
            {
                "@id": "elem-provider",
                "amount": 1.0,
                "flow": {"@id": "flow-provider", "name": "Protected"},
                "unit": {"name": "kg"},
                "provider": {"@id": "provider-1"},
            },
        ],
    }
    flows = {
        "flow-cover": FlowInfo("flow-cover", "Cover", "ELEMENTARY_FLOW", "air/urban air", True, {}),
        "flow-provider": FlowInfo("flow-provider", "Protected", "ELEMENTARY_FLOW", "air/urban air", True, {}),
    }
    category = ImpactCategory(
        "cat",
        "Climate",
        "m1",
        "IPCC",
        "",
        "",
        "kg",
        {
            "flow-cover": CharacterizationFactor("flow-cover", 0.95, "kg", {}),
            "flow-provider": CharacterizationFactor("flow-provider", 0.05, "kg", {}),
        },
        {},
    )
    result = reduce_process(
        process_data=process,
        flow_lookup=flows,
        categories=[category],
        tau=0.95,
        uncharacterised_policy="drop",
        strict_units=True,
        tol=1e-12,
        database_name="db",
    )
    assert result.process_row["positive_cover_ok"] is True
    assert result.process_row["negative_cover_ok"] is True
    assert result.process_row["min_positive_coverage"] == 1.0
    assert result.process_row["min_negative_coverage"] == 1.0
    assert result.process_row["n_active_positive_categories"] == 1
    assert result.process_row["n_active_negative_categories"] == 0
