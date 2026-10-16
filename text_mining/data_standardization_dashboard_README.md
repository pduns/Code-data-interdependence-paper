# Data Source Standardization Dashboard

This is a draft local dashboard for reviewing LLM-detected data source names and mapping them to canonical data holders. It shows the current label, LLM reasoning, all labels returned for the same document, and the `file_name`.

## Launch

From the repository root, after setting `DATA_INTERDEPENDENCIES_DATA_ROOT`:

```powershell
python text_mining\data_standardization_dashboard.py
```

Then open:

```text
http://127.0.0.1:8765
```

Use `Ctrl+C` in the terminal to stop the server.

## Inputs

- `llm_labelled_data/compiled/llm_labelled_data_sources_long.csv`

## Outputs

- `processed_data/data_holder_dictionary.csv`
- `processed_data/data_source_standardization_decisions.csv`

The dictionary starts empty unless an existing CSV is already present. New holder entries are saved immediately. Review decisions are upserted by stable `review_id`, so saving the same source row again updates the previous decision. Existing holders are suggested when their names match the start of the current label or the typed holder search text.

## Current Decision Types

- `data_resource`: the encountered label is a resource and should be mapped to a holder.
- `data_holder`: the encountered label is itself the holder/provider.
- `false_positive`: the LLM label should not be retained as a valid external digital data source/holder.
- `manual_unresolved`: valid data use is likely, but the label/holder cannot be resolved yet.
- `needs_review`: defer for later adjudication.
