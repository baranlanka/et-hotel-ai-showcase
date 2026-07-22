# Observability stack

A self-contained, runnable slice of the production telemetry pipeline: the
OpenTelemetry / Grafana **LGTM**-style triad the platform runs to trace, log and
measure ~10 services from a single pane of glass. It is sanitized (no real
hosts, domains, credentials, or business service names) but wired the same way
production is.

```
app + workers ──OTLP──▶ OpenTelemetry Collector ──▶ Tempo   (traces)
                                                └──▶ Loki    (logs)
                                                └──▶ Prometheus exporter (metrics)
host log files ──▶ Promtail ──▶ Loki
                    Tempo · Loki · Prometheus ──▶ Grafana (dashboards + correlations)
```

## Run it

```bash
cd infra/observability
docker compose up -d
open http://localhost:3000        # Grafana — anonymous access, dashboards pre-loaded
```

Data sources (Loki, Tempo, Prometheus) are provisioned and correlated
(logs↔traces↔metrics) out of the box. Panels are idle until something sends
telemetry — point an app's OTLP exporter at `otel-collector:4317` (gRPC) or
`:4318` (HTTP), and tail its logs by setting `LOG_DIR` for Promtail. Credentials
are env placeholders with safe local defaults (`GRAFANA_ADMIN_PASSWORD`,
`GRAFANA_SECRET_KEY`); override them via a local `.env` for anything non-local.

Ports: Grafana `3000`, Prometheus `9090`, Tempo `3200`, Loki `3100`,
Collector OTLP `4317`/`4318` and Prometheus exporter `8889`.

## What's inside

| Component | Config | Role |
|---|---|---|
| OpenTelemetry Collector | `otel-collector/collector-config.yaml` | Single OTLP ingest; fans traces→Tempo, logs→Loki, metrics→Prometheus exporter |
| Tempo | `tempo/tempo.yaml` | Trace store; metrics-generator emits service-graph + span-metrics |
| Loki | `loki/loki-config.yaml` | Log store (filesystem, 7-day retention) |
| Promtail | `promtail/promtail-config.yaml` | Tails structured JSON logs, promotes `level`/`service_name`/`trace_id` to labels |
| Prometheus | `prometheus/prometheus.yml` | Scrapes the collector, Tempo, and generic `app`/`worker` jobs |
| Grafana | `grafana/grafana.ini` + `grafana/provisioning/` | Dashboards + datasource provisioning |

## Dashboards

- **Platform Overview** (home) — service up/down, recent app errors, active workflows.
- **Application Overview** — API request rate, success rate, search p95 latency, worker status, recent errors.
- **OpenTelemetry Collector Monitoring** — pipeline throughput, receiver success rate, export queue depth, failed exports.
- **Tempo Distributed Tracing** — trace ingestion rate, query latency, service map (node graph), top services by span count, error traces.

The datasources are cross-linked: a log line's `trace_id` jumps to its trace in
Tempo; a trace jumps back to its logs in Loki and to span-metrics in Prometheus.
