# Data card: CARE-GNN YelpChi (Track A)

| Field | Value |
|---|---|
| File | `data/raw/care/YelpChi.mat` |
| SHA256 | `fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42` |
| Source (used) | github.com/YingtongDou/CARE-GNN, commit `a64ff7523e187a24251f7ca88435d2c9d8f7dcd9`, `data/YelpChi.zip` |
| Source (attempted first) | Kaggle mirror `hunhthanhdn/yelpchi-dataset-gat` — **rejected**: preprocessed PyG `.pt` with a single concatenated 7.69M-edge graph; per-relation identity (spec 6.1 contract) is lost and the bundle ships foreign splits. Rejection is loud in `scripts/get_data.py`, never silent. |
| Original provenance | YelpCHI dataset (Shebuti Rayana's YelpChi); processed by the CARE-GNN authors into MATLAB matrices |
| Nodes | 45,954 reviews |
| Features | 32 processed numeric features (dense float32 after loading) |
| Labels | 1 = spammy (6,677), 0 = benign (39,277); **proxy labels = Yelp's filtering decision**, not verified fraud |
| Relations | `net_rur` (98,630 edges), `net_rtr` (1,147,232), `net_rsr` (6,805,486); `homo` = union (7,693,958) |
| Access | public download via the CARE-GNN repository |
| License / redistribution | underlying reviews are Yelp content; the processed benchmark carries the repository's terms. Do not redistribute raw data. |

## What this data CANNOT support

- No review text, no timestamps, no reviewer identities, no business identities exist at this processing level. Track A never invents them, and Track A results must not be described as text/timeline fraud detection.
- Relations were formed over the whole source period: the transductive split used here is a benchmark protocol, **not** future-event prediction.
- Labels are approximate (platform filtering). Known errors/corrections are documented in the CARE-GNN README and should be read before any reproduction claim against the original paper.
