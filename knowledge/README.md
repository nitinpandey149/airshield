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
the "assistant layer" section of `docs/architecture.md` for why that boundary is
enforced.

## Sources

Only material from official agencies and reputable public-health bodies is
included:

| Source | What it provides |
| --- | --- |
| US EPA | PM2.5 basics, health effects, AQI calculation, ozone, NO2, VOCs, particle size |
| US EPA AirNow | AQI categories and technical reporting, particle-pollution guidance, outdoor-activity guidance for schools |
| WHO | Ambient air quality fact sheet, global air quality guidelines, physical activity guidance |
| American Lung Association | Particle pollution and health, exercising outdoors |

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
make knowledge          # refresh sources from the official URLs, then index
make knowledge-fetch    # fetch/refresh the source text only
make knowledge-offline  # rebuild the index from the already-fetched sources
```

Or directly:

```bash
python knowledge/fetch_sources.py             # fetch anything missing
python knowledge/fetch_sources.py --refresh   # re-fetch everything
python knowledge/build_index.py               # build the vector index
```

The pipeline is:

```
manifest.json -> fetch (cached) -> extract text -> clean -> chunk -> embed -> index
```

Each chunk carries the metadata of its parent document, so a retrieved chunk can
always be traced back to a real source with a real URL.

The extracted text under `sources/` is committed, so the assistant works offline.
The vector index under `index/` is a build output and is gitignored; rebuild it
with `make knowledge-offline`.

## Licensing and attribution

The extracts remain the property of their publishers and are reproduced here for
non-commercial research and demonstration, with attribution. See each document's
`licence` field. This project does not claim ownership of any source material.
