# Text mining and analysis

This stage starts with the ModernBERT-filtered publication texts and produces the standardized resource-holder data, dependence measures, figures and final Supplementary Data workbook.

The principal sequence is:

1. `1.create_human_coding_sample_from_filtered_texts.ipynb`
2. `2.develop_and_evaluate_llm_extraction_prompt.ipynb`
3. `3.apply_llm_extraction_to_filtered_corpus.ipynb`
4. `4.compile_llm_labelled_corpus_results.ipynb`
5. `5.standardize_extracted_data_source_names.ipynb`
6. `6.1.re-build_with_cleaned_names_and_merge_standardized_sources_with_bibliometric_data.ipynb`
7. `6.2. Transition_to_data_holder_logic_workbook.ipynb`
8. `6.3. Overwrite_old_metadata_with_new_and_visualizaiton_names.ipynb`
9. `7.summarize_data_source_counts_and_fields.ipynb`
10. `8.create_source_user_country_networks.ipynb`
11. `8.1.create_data_holder_country_networks.ipynb`
12. `8.2.create_economic_group_provider_heatmap.ipynb`
13. `9.interdependence_analysis_over_time.ipynb`
14. `10.prepare_final_supplementary_materials.ipynb`

`9.interdependence_analysis_over_time_through_2024.ipynb` contains the extended temporal diagnostic used to assess the decline in publication coverage after 2021.

The extraction prompt is stored as a standalone, versioned text file in `prompts/`. The notebooks read it from there rather than from an archived development notebook.

Additional command-line utilities support batch retrieval, fast ModernBERT filtering, source-standardization review, folded-network generation and final figure construction. All paths are resolved through the repository's `project_config.py` or explicit command-line arguments.
