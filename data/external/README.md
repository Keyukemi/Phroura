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

