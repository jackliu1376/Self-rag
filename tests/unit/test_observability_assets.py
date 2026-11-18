import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OBS = ROOT / "deploy" / "observability"


def test_dashboard_is_valid_json_and_uses_every_metric() -> None:
    dashboard = json.loads(
        (OBS / "grafana" / "dashboards" / "self-rag.json").read_text(encoding="utf-8")
    )
    expressions = json.dumps(dashboard, ensure_ascii=False)
    for metric in (
        "cs_requests_total",
        "cs_request_latency_seconds_bucket",
        "cs_stage_latency_seconds_bucket",
        "cs_routing_total",
        "cs_tool_calls_total",
        "cs_tokens_total",
        "cs_hallucination_flags_total",
    ):
        assert metric in expressions
    assert dashboard["uid"] == "self-rag"


def test_prometheus_scrapes_both_api_instances() -> None:
    config = yaml.safe_load((OBS / "prometheus.yml").read_text(encoding="utf-8"))
    targets = config["scrape_configs"][0]["static_configs"][0]["targets"]
    assert set(targets) == {"api-1:8000", "api-2:8000"}


def test_compose_overlay_provisions_grafana() -> None:
    config = yaml.safe_load((ROOT / "docker-compose.observability.yml").read_text(encoding="utf-8"))
    services = config["services"]
    assert {"prometheus", "grafana"} <= set(services)
    grafana_volumes = services["grafana"]["volumes"]
    assert any("grafana/provisioning" in volume for volume in grafana_volumes)
    assert any("grafana/dashboards" in volume for volume in grafana_volumes)
