# External Dataset Notes

## pirocheto/phishing-url

Source:

```text
https://huggingface.co/datasets/pirocheto/phishing-url
```

Local files:

```text
data/external/pirocheto_phishing_url_train.parquet
data/external/pirocheto_phishing_url_test.parquet
```

Purpose:

- external adversarial-style robustness testing
- selection of phishing URLs with obfuscation or evasion indicators
- balanced external validation using both phishing and legitimate URLs
- multi-source retraining using the train split

The external dataset's precomputed feature columns are not used for Phroura prediction. They are used only to select adversarial-style rows. Phroura re-runs its own lexical feature extractor on the raw URL strings before prediction.

Indicators used for adversarial-style filtering:

- `punycode`
- `shortening_service`
- `nb_at`
- `ip`
- `abnormal_subdomain`
- `brand_in_subdomain`
- `brand_in_path`
- `random_domain`

The current Phase 3 robustness artifact uses the dataset's test split and keeps phishing rows with at least one of these indicators.

The current Phase 3 external validation artifact uses the full test split:

```text
3772 total rows
1886 phishing rows
1886 legitimate rows
```

The current Phase 3 multi-source model uses:

```text
URL-Phish original training data
pirocheto/phishing-url train split
```

It is evaluated on the held-out `pirocheto/phishing-url` test split.

## LegitPhish Dataset

Source:

```text
https://data.mendeley.com/datasets/hx4m73v2sf/2
```

Local files:

```text
data/legitPhish.csv
data/external/legitphish_normalized.csv
```

Purpose:

- final independent external validation
- third-source generalization check after multi-source retraining
- comparison between the original Random Forest, the multi-source Random Forest, and the heuristic baseline

The original LegitPhish file contains precomputed URL feature columns. Phroura does not use those feature columns for prediction. Phroura uses only the raw `URL` strings and re-runs its own lexical feature extractor before scoring.

Important label mapping:

```text
LegitPhish ClassLabel 0 = phishing
LegitPhish ClassLabel 1 = legitimate
Phroura label 1 = phishing
Phroura label 0 = legitimate
```

The normalized file therefore converts:

```text
ClassLabel 0 -> label 1
ClassLabel 1 -> label 0
```

The current Phase 3 LegitPhish validation artifact uses:

```text
101218 usable rows
63678 phishing rows
37540 legitimate rows
1 row dropped because its label was missing
```
