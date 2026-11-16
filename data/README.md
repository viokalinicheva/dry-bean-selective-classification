# Dataset provenance

This project includes the unchanged UCI **Dry Bean** archive under Creative Commons Attribution 4.0 International (CC BY 4.0).

- Source: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- DOI: https://doi.org/10.24432/C50S4B
- Download: https://archive.ics.uci.edu/static/public/602/dry%2Bbean%2Bdataset.zip
- Retrieved: November 3, 2026; documentation/license checked November 9, 2026.
- License: https://creativecommons.org/licenses/by/4.0/
- Archive SHA-256: `0a64eff5be87f48c3dbbfc0a12a56c5d5b5167ef8e61cd45d69b3e7c7130c06f`
- Parsed ARFF SHA-256: `b2a4a76a2aedfb8ed415adfc1bfc70b5f202cb00cb72e500766e364c14834014`

Citation: *Dry bean* [Data set]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C50S4B

The source archive's README also requests citation of Köklü and Özkan's 2020 article: https://doi.org/10.1016/j.compag.2020.105507.

The analysis parses the ARFF directly from the ZIP. It preserves every source row, predictor, and label; no imputation, joins, unit conversion, exclusion, or relabeling is performed. Exact-vector group IDs and split assignments are derived metadata, never model predictors. Learned scaling is fitted within training only.

The dataset license is separate from the MIT license for original project code. It does not establish reuse rights for the journal article or original photographs. No original seed photographs are supplied here. See [ATTRIBUTION.txt](../ATTRIBUTION.txt) for complete notices.
