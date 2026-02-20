# Nemesis — Execution Order & Work Packages

> Strukturierter Fahrplan: 82 Tasks → 30 Arbeitspakete → 4 Phasen

---

## Dependency Graph (Pläne)

```
Phase 1        Phase 2 (parallel)       Phase 3          Phase 4         Phase 5 (parallel)
─────────────────────────────────────────────────────────────────────────────────────────────

01-project  ──┬── 02-rust-parser  ──┐
              ├── 03-graph-layer  ──┼── 05-indexing  ── 06-mcp-server ─┬── 07-memory-system
              └── 04-vector-store ──┘                                  └── 08-cli-hooks
```

---

## Alle Arbeitspakete (30 Stück)

### Plan 01 — Project Setup (10 Tasks → 3 Pakete)

| Paket | Tasks | Beschreibung | Abhängig von |
|-------|-------|-------------|--------------|
| **A1** | 1, 2, 3 | Git + pyproject.toml + Package-Struktur | — |
| **A2** | 4, 5, 7, 10 | Config + CLI + Fixtures + Integrationstest | A1 |
| **A3** | 6, 8, 9 | Rust-Crate + GitHub Actions + Ruff | A1 |

### Plan 02 — Rust Parser (12 Tasks → 4 Pakete)

| Paket | Tasks | Beschreibung | Abhängig von |
|-------|-------|-------------|--------------|
| **B1** | 1, 2, 3 | Rust Data Models + Grammars + Parser | A1 |
| **B2** | 4, 5, 6 | Extractor + PyO3 Bindings + Build | B1 |
| **B3** | 7, 8, 9, 10 | Python Bridge + Public API + Fixtures | B2 |
| **B4** | 11, 12 | Integration Tests + Lint | B3 |

### Plan 03 — Graph Layer (10 Tasks → 4 Pakete)

| Paket | Tasks | Beschreibung | Abhängig von |
|-------|-------|-------------|--------------|
| **C1** | 1, 2 | GraphAdapter Protocol + Kuzu Init | A1 |
| **C2** | 3, 4, 5, 6 | Kuzu CRUD + Edges + File Ops + Traversal | C1 |
| **C3** | 7, 9 | Neo4j Adapter + Memory Nodes | C1 |
| **C4** | 8, 10 | Factory + Integration Test | C2 |

### Plan 04 — Vector Store (8 Tasks → 3 Pakete)

| Paket | Tasks | Beschreibung | Abhängig von |
|-------|-------|-------------|--------------|
| **D1** | 1, 2, 3 | Embedding Protocol + OpenAI + Local | A1 |
| **D2** | 4, 5, 6 | VectorStore Init + Add + Search | D1 |
| **D3** | 7, 8 | Delete + Pipeline Helpers | D2 |

### Plan 05 — Indexing Pipeline (10 Tasks → 5 Pakete)

| Paket | Tasks | Beschreibung | Abhängig von |
|-------|-------|-------------|--------------|
| **E1** | 1, 2 | Datenmodelle + Token-Counting | A1 |
| **E2** | 3, 4 | Chunker (klein + groß) | E1 |
| **E3** | 5, 6 | Delta Detection + Deletion | E1 |
| **E4** | 7, 8 | Single File Index + Reindex | E2, E3, B3 |
| **E5** | 9, 10 | Full Project + Delta Update | E4 |

### Plan 06 — MCP Server (12 Tasks → 4 Pakete)

| Paket | Tasks | Beschreibung | Abhängig von |
|-------|-------|-------------|--------------|
| **F1** | 1, 2 | Pydantic Models + NemesisEngine | A1 |
| **F2** | 3, 4, 5, 6 | Server + Code Intelligence Tools | F1, C2, D2 |
| **F3** | 7, 8, 9 | Memory + Index Tools | F1, C2, E5 |
| **F4** | 10, 11, 12 | Smart Context + Dispatch + Entry Point | F2, F3 |

### Plan 07 — Memory System (10 Tasks → 5 Pakete)

| Paket | Tasks | Beschreibung | Abhängig von |
|-------|-------|-------------|--------------|
| **G1** | 1 | Pydantic Data Models | A1 |
| **G2** | 2, 3 | RulesManager (Basic + Advanced) | G1, C2 |
| **G3** | 4, 5 | DecisionsManager (Basic + Advanced) | G1, C2 |
| **G4** | 6, 7 | ConventionManager + SessionContext | G2, G3 |
| **G5** | 8, 9, 10 | AutoLearn + Integration | G4 |

### Plan 08 — CLI & Hooks (10 Tasks → 3 Pakete)

| Paket | Tasks | Beschreibung | Abhängig von |
|-------|-------|-------------|--------------|
| **H1** | 1, 2, 3 | CLI: init + index/status + query/rule | F4 |
| **H2** | 4, 5, 6 | File Watcher + CLI watch | H1, E5 |
| **H3** | 7, 8, 9, 10 | Hooks + Config Generator | H2, G5 |

---

## Empfohlene Execution-Reihenfolge

### Sprint 1: Foundation

```
A1 → A2 → A3
```
- 10 Tasks, ~37 Tests
- Ergebnis: Lauffähiges Projekt mit CI

### Sprint 2: Core Layer (parallel)

```
B1 → B2 → B3 → B4     (Rust Parser)
C1 → C2 → C3 → C4     (Graph Layer)  ← parallel
D1 → D2 → D3           (Vector Store) ← parallel
```
- 30 Tasks, ~214 Tests
- B, C, D sind voneinander unabhängig → parallel möglich
- Ergebnis: Parser, Graph DB, Vector DB einzeln funktional

### Sprint 3: Integration

```
E1 → E2 ──┐
E1 → E3 ──┼→ E4 → E5  (Indexing Pipeline)
           │
G1 → G2 ──┤            (Memory System — kann parallel starten)
G1 → G3 ──┼→ G4 → G5
```
- 20 Tasks, ~97 Tests
- E2/E3 parallel, G2/G3 parallel
- Ergebnis: Vollständige Pipeline + Memory

### Sprint 4: User-Facing

```
F1 → F2 → F3 → F4      (MCP Server)
H1 → H2 → H3           (CLI & Hooks) ← nach F4
```
- 22 Tasks, ~153 Tests
- Ergebnis: Fertiges Produkt

---

## Zusammenfassung

| Sprint | Pakete | Tasks | Tests | Parallelisierbar |
|--------|--------|-------|-------|------------------|
| 1 Foundation | A1-A3 | 10 | ~37 | Nein (linear) |
| 2 Core Layer | B1-B4, C1-C4, D1-D3 | 30 | ~214 | Ja (3 Tracks) |
| 3 Integration | E1-E5, G1-G5 | 20 | ~97 | Ja (2 Tracks) |
| 4 User-Facing | F1-F4, H1-H3 | 22 | ~153 | Teilweise |
| **Gesamt** | **30 Pakete** | **82** | **~501** | |
