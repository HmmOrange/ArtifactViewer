# Failed-case analysis: Hitab and mulhi

Only failed outcomes are included. A `wrong case` is a completed execution whose answer remains wrong after the LLM judge. A `fail case` is either a wrong case or a source execution error.

| Benchmark | Solution | Wrong | Misalignment | Misinterpretation | All fail cases | Failed code |
|---|---:|---:|---:|---:|---:|---:|
| Hitab | GraphOtter | 114 | 114 / 114 (100.00%) | 0 / 114 (0.00%) | 220 | 106 / 220 (48.18%) |
| mulhi | GraphOtter | 142 | 73 / 142 (51.41%) | 69 / 142 (48.59%) | 145 | 3 / 145 (2.07%) |
| Hitab | ST-raptor | 21 | 8 / 21 (38.10%) | 13 / 21 (61.90%) | 57 | 36 / 57 (63.16%) |
| mulhi | ST-raptor | 33 | 19 / 33 (57.58%) | 14 / 33 (42.42%) | 199 | 166 / 199 (83.42%) |
| Hitab | SpreadsheetAgent | 187 | 187 / 187 (100.00%) | 0 / 187 (0.00%) | 187 | 0 / 187 (0.00%) |
| mulhi | SpreadsheetAgent | 111 | 111 / 111 (100.00%) | 0 / 111 (0.00%) | 125 | 14 / 125 (11.20%) |

## Audit files

- `failure_modes.xlsx`: summary chart, filters, conditional formatting, and one detail sheet per benchmark/solution.
- `all_failed_cases.csv`: every failed case with X/Z previews, evidence, W error, Y/Y*, and artifact paths.
- `report.html`: filterable visual report with a filtered-CSV download button.
- `cases/*.csv`: one CSV per benchmark/solution.

## Classification rules

- Misalignment: W completed, Y differs from Y*, and Z both faithfully represents X and covers every gold-relevant table.
- Misinterpretation: W completed, Y differs from Y*, and Z is missing, structurally inconsistent with X, selects the wrong table, or omits a gold-relevant MulHi table.
- Failed code: W is incomplete or failed to execute. Its denominator is all failed outcomes (`wrong + sourceErrors`).
- MulHi expected table indices come from the first component of each gold `table_evidence` cell ID. For GraphOtter/ST-Raptor, the selected table set must equal that expected set; selecting only one of two required tables is a mismatch.
- SpreadsheetAgent used an explicit workbook-Markdown fallback because the official extraction services were unavailable. The fallback counts as Z because the saved workflow used it; every represented sheet and non-empty X cell is checked.

Total detailed failed cases: 933.
