# Dataset curation

These notebooks build the publication corpus and the ModernBERT-prefiltered text collection used by the text-mining stage.

Run them in order:

1. `1.get_texts_linked_to_keywords_from_lens.ipynb` constructs the indicator searches and retrieves bibliographic records through the Lens API.
2. `2.merge_literature_lists.ipynb` combines the target-specific retrieval files.
3. `3.filter_literature_based_on_subject_fields.ipynb` applies subject-field and bibliographic filters.
4. `4.paper_evaluation.ipynb` performs corpus diagnostics and additional screening.
5. `5.download_pdfs.ipynb` retrieves available open-access PDFs.
6. `6.parse_pdfs_with_grobid.ipynb` extracts relevant text through a local GROBID service.
7. `7.create_training_sample_and_paragraph_coding_sheet.ipynb` draws the reproducible manual-coding sample.
8. `8.train_modernbert_prefilter.ipynb` trains and evaluates the binary paragraph prefilter.
9. `9.filter_parsed_texts_with_modernbert_prefilter.ipynb` applies the selected threshold to the parsed corpus.

Helper modules are stored in `functions/`. Machine-specific data locations and credentials have been replaced by the shared configuration described in the repository-level README.
