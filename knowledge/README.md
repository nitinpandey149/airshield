# AirShield knowledge base

Trusted environmental and public-health material used by the **Ask AirShield**
assistant. Nothing here is generated: every document is a verbatim extract of an
official or peer-reviewed source, and every document records where it came from.

## Why this exists

The assistant answers *knowledge* questions ("What is PM2.5?", "What does AQI
mean?") by retrieving from this corpus. It never answers those from the model's
memory, and it never uses this corpus to produce a number.

The numerical side of AirShield — PM2.5 forecasts, AQI, exposure scores, route
rankings, best time windows — is produced by the XGBoost model and the
deterministic exposure engine in `core/`. RAG is not involved in any of it. See
`docs/rag.md` for why that boundary is enforced.

## Sources

Only material from official agencies and reputable public-health/scientific
bodies is included:

| Source | What it provides |
| --- | --- |
| US EPA | PM2.5/PM10 basics, health effects, AQI calculation, ozone, NO2, particle size |
| US EPA AirNow | AQI categories, the six pollutants, who is at risk |
| WHO | Ambient air quality guidelines, health effects, physical activity guidance |
| US CDC | Particle pollution and heart/lung health |
| US NIH / PubMed Central | Peer-reviewed studies on exercise and inhaled pollutant dose |

Blogs, SEO content and unattributed summaries are deliberately excluded.

## Layout

```
knowledge/
├── manifest.json         # index of documents: id, category, source, url, licence
├── sources/              # verbatim extracted text, one file per document
├── build_index.py        # ingestion: extract -> clean -> chunk -> embed -> store
└── README.md
```

Documents are grouped by category: `air_quality`, `pollutants`, `exposure`,
`outdoor_activity`, `AQI`, `environmental_guidance`.

## Ingestion

```bash
python knowledge/build_index.py            # fetch sources, build the index
python knowledge/build_index.py --offline  # rebuild from the bundled cache
```

The pipeline is:

```
manifest.json -> fetch (cached) -> extract text -> clean -> chunk -> embed -> index
```

Each chunk carries the metadata of its parent document, so a retrieved chunk can
always be traced back to a real source with a real URL.

## Licensing and attribution

The extracts remain the property of their publishers and are reproduced here for
non-commercial research and demonstration, with attribution. See each document's
`licence` field. This project does not claim ownership of any source material.
