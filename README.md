# Data harmonisation pipeline for MedNorm corpus
The data harmonisation pipeline combines instances from various datasets 
and provides consistent simultaneous mappings to both MedDRA and SNOMED-CT terminologies.
It also generates a corpus graph and cross-terminology concept embeddings.
### A corpus and embeddings for cross-terminology medical concept normalisation
The corpus and embeddings are available here: https://doi.org/10.17632/b9x7xxb9sz.1



### Step-by-step guide
##### 1. Combine datasets
```
python dataset.py combine --config=~/research/data/mednorm/datasets.yaml \
    --output=~/research/data/mednorm/generate/mednorm_raw.tsv
```
```
Processing datasets:
        [*] CADEC
          1250 file pairs
          860 skipped lines
          7009 converted lines
        [*] TwADR-L
          5074 converted lines
        [*] TwiMed-PubMed
          1579 skipped lines
          1142 converted lines
        [*] TwiMed-Twitter
          1594 skipped lines
          827 converted lines
        [*] SMM4H2017-train
          6650 converted lines
        [*] SMM4H2017-test
          2499 converted lines
        [*] TAC2017_ADR
          7045 converted lines
File /home/mbelousov/research/data/mednorm/generate/mednorm_raw.tsv saved.
Done. 30246 lines combined.
```
##### 2. Build initial corpus graph representation
```
python dataset.py build_graph \
    --dataset=~/research/data/mednorm/generate/mednorm_raw.tsv \
    --output=~/research/data/mednorm/generate/mednorm_raw.graph \
    --mysql-host=127.0.0.1 --mysql-user=root --mysql-db=umls \
    --rules=./resources/rules.tsv
```

##### 3. Build concept embeddings model
```
python dataset.py build_embeddings \
    --graph=~/research/data/mednorm/generate/mednorm_raw.graph \
    --output=~/research/data/mednorm/embeddings/mednorm_raw_10n_40l_5w_64dim \
    --mode=deepwalk \
    --n=10 --length=40 --dim=64 --w=5 --seed=42
```

##### 4. Identify potential annotators errors
```
python dataset.py unrelated_annotations \
    --graph=~/research/data/mednorm/generate/mednorm_raw.graph \
    --output=~/research/data/mednorm/generate/analysis/mednorm \
    --dthresh=1
```

```
python dataset.py ambiguous_tokens \
    --dataset=~/research/data/mednorm/generate/mednorm_raw.tsv \
    --graph=~/research/data/mednorm/generate/mednorm_raw.graph \
    --embeddings=~/research/data/mednorm/embeddings/mednorm_raw_10n_40l_5w_64dim.bin \
    --output=~/research/data/mednorm/generate/analysis/mednorm_amb_tokens.tsv
```

##### 5. Correct annotation errors
```
python dataset.py human_correct \
    --dataset=~/research/data/mednorm/generate/mednorm_raw.tsv \
    --corrections=./resources/human_corrections.tsv \
    --output=~/research/data/mednorm/generate/mednorm_corrected.tsv
```

##### 6. Build final graph representation
```
python dataset.py build_graph \
    --dataset=~/research/data/mednorm/generate/mednorm_corrected.tsv \
    --output=~/research/data/mednorm/generate/mednorm_corrected.graph \
    --mysql-host=127.0.0.1 --mysql-user=root --mysql-db=umls \
    --rules=./resources/rules.tsv
```

##### 7. Generate TSV dataset
```
python dataset.py tsv \
    --graph=~/research/data/mednorm/generate/mednorm_corrected.graph \
    --dataset=~/research/data/mednorm/generate/mednorm_corrected.tsv \
    --output=~/research/data/mednorm/generate/mednorm_mapped_draft.tsv \
    --parallel \
    --non_empty
```
```
Initial: 30246
After terminology filtering: 27979
Rows: 27979
```

##### 8. Resolve phrase duplicates
```
python dataset.py resolve_dups \
    --dataset=~/research/data/mednorm/generate/mednorm_mapped_draft.tsv \
    --target_col=mapped_meddra_codes,mapped_sct_ids \
    --output=~/research/data/mednorm/generate/mednorm_mapped.tsv \
    --keep_ratio=0.1 \
    --ignore_case
```
```
Conflict phrases: 397 (out of 10572 unique)
Rows changed: 6667 (out of 27979)
```
##### 9. Reduce to single label

```
python dataset.py reduce \
    --dataset=~/research/data/mednorm/generate/mednorm_mapped.tsv \
    --output=~/research/data/mednorm/generate/mednorm_full.tsv \
    --label_col=mapped_meddra_codes,mapped_sct_ids \
    --embeddings=~/research/data/mednorm/embeddings/mednorm_raw_10n_40l_5w_64dim.bin \
    --graph=~/research/data/mednorm/generate/mednorm_corrected.graph \
    --source_col=meddra_code,sct_id,umls_cui \
    --ignore_case
```
```
Instances: 27979
mapped_meddra_codes	orig: 2242 reduced: 2080 single
mapped_sct_ids	orig: 2752 reduced: 2100 single
Conflict phrases: 207 (out of 10572 unique)
COL: single_mapped_meddra_codes, 0 duplicates, 2062 labels
COL: single_mapped_sct_ids, 0 duplicates, 2089 labels
```


##### 10. Filtering

```
python dataset.py  filter \
    --dataset=~/research/data/mednorm/generate/mednorm_full.tsv \
    --original=CADEC,TwiMed-Twitter,SMM4H2017-train,SMM4H2017-test,TwADR-L,TAC2017_ADR,TwiMed-PubMed \
    --output=~/research/data/mednorm/mednorm_full.tsv
```
## Citation
Please cite the following paper:
```
@inproceedings{belousov2019mednorm,
  title={MedNorm: A Corpus and Embeddings for Cross-terminology Medical Concept Normalisation},
  author={Belousov, Maksim and Dixon, William G and Nenadic, Goran},
  booktitle={},
  pages={},
  year={2019}
}
```