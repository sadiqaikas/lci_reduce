from copy import deepcopy

from lci_reduce.models import CharacterizationFactor, FlowInfo, ImpactCategory
from lci_reduce.reducer import reduce_process


def test_jsonld_process_editing():
    process = {
        "@id": "process-1",
        "name": "Toy Process",
        "location": {"name": "GB"},
        "exchanges": [
            {
                "@id": "product",
                "amount": 1.0,
                "flow": {"@id": "flow-product", "name": "Product"},
                "unit": {"name": "kg"},
                "quantitativeReference": True,
            },
            {
                "@id": "elem-keep",
                "amount": 10.0,
                "flow": {"@id": "flow-co2", "name": "CO2"},
                "unit": {"name": "kg"},
            },
            {
                "@id": "elem-drop",
                "amount": 0.1,
                "flow": {"@id": "flow-ch4", "name": "CH4"},
                "unit": {"name": "kg"},
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
    original = deepcopy(process)
    flows = {
        "flow-product": FlowInfo("flow-product", "Product", "PRODUCT_FLOW", "", False, {}),
        "flow-co2": FlowInfo("flow-co2", "CO2", "ELEMENTARY_FLOW", "air/urban air", True, {}),
        "flow-ch4": FlowInfo("flow-ch4", "CH4", "ELEMENTARY_FLOW", "air/urban air", True, {}),
        "flow-tech": FlowInfo("flow-tech", "Input", "PRODUCT_FLOW", "", False, {}),
    }
    category = ImpactCategory(
        category_id="cat-1",
        name="Climate",
        method_id="m1",
        method_name="IPCC",
        path="",
        metadata_text="",
        reference_unit="kg",
        factors={
            "flow-co2": CharacterizationFactor("flow-co2", 10.0, "kg", {}),
            "flow-ch4": CharacterizationFactor("flow-ch4", 0.1, "kg", {}),
        },
        raw={},
    )
    result = reduce_process(
        process_data=process,
        flow_lookup=flows,
        categories=[category],
        tau=0.95,
        uncharacterised_policy="keep",
        strict_units=True,
        tol=1e-12,
        database_name="db",
    )
    exchange_ids = [exchange["@id"] for exchange in result.reduced_process["exchanges"]]
    assert "elem-drop" not in exchange_ids
    assert "elem-keep" in exchange_ids
    assert "product" in exchange_ids
    assert "tech" in exchange_ids
    assert result.reduced_process["location"] == original["location"]
