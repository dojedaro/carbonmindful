# Build log: ten improvement passes (2026-11-01)

Each pass had one focus and was checked by running something. Findings are listed as they
happened, including the defects each pass exposed.

| Pass | Focus | What it found or changed | Check |
|---|---|---|---|
| 1 | Calculation core, bundle format | One chain with PUE applied once; synthetic demo data marked as such | 31 tests pass; chart palette passes the colour-blind validator |
| 2 | Registry, charts, report, first app | Factors carry value, unit, source and year; worked example of the calculation | App builds |
| 3 | First real run | 37 tests pass, yet screens showed `1.60e+3` labels, crowded axes, cut-off names, raw column names | Screenshots of every tab |
| 4 | Result integrity | Comparison tab would have averaged 3-epoch with 20-epoch runs, laptop with Colab runs, and synthetic with real runs; added a check that epoch energy matches the run total | Tests added |
| 5 | Untrusted input | 18 of 18 new attack and bad-data tests failed first: corrupt and encrypted zips crashed; markup in fields, NaN and infinite values were accepted; all fixed | 58 tests pass |
| 6 | Chart readability | Each chart picks mg, g or kg; tick limits; full labels; long series thinned | 79 tests pass |
| 7 | Layout | Typed settings, single notice banner, bordered tiles, readable tables, a data table with the charts | Screenshots reviewed |
| 8 | Test strength | Coverage was 97%, yet 2 of 12 deliberate bugs went unnoticed, including PUE dropped from epoch values; tests strengthened; real browser upload test added | 12 of 12 bugs caught |
| 9 | Engineering standards | Strict type checking and lint found 6 type errors and 8 lint issues; pinned versions, README with bundle format | `mypy --strict`, `ruff` clean |
| 10 | Critical review | 11 new tests failed first: zero-energy crash, huge-number crash, nested-JSON crash, fractional epochs, seed range, test and validation accuracy averaged together, repeated seeds unflagged, markup in file names, missing version and factor sources in exports; all fixed | Full suite below |

## Final state

- 96 tests pass, with 97% line coverage.
- `mypy --strict` and `ruff` are clean.
- The browser upload test passes: a known bundle displays the hand-calculated 10 Wh and 9.8 g CO₂e.
- The screenshot check shows no page exceptions or console errors on any tab.
- 12 of 12 deliberate bugs are caught. These 12 were chosen by hand, so this is not an exhaustive mutation test.

## Known limits

- It has only been run on synthetic and test bundles, because the measurement script that writes real bundles does not exist yet.
- Grid factors stay marked provisional until checked against Ember's own file.
- It is a local, single-user tool with no login. Do not expose it publicly without adding access control.
- The theme is light only.
