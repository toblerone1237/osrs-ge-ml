# Instructions for Codex

- ML code lives in `ml/` and uses R2 bucket `osrs-ge-raw`.
- 5m worker (`workers/osrs-ge-5m`) writes 5m snapshots and latest data to R2.
- Daily worker (`workers/osrs-ge-daily`) writes daily/latest snapshots + mapping to R2.
- UI worker (`workers/osrs-ge-ui`) serves HTML and endpoints:
  - GET /signals -> signals/latest.json in R2
  - GET /daily   -> daily/latest.json in R2
  - GET /price-series?item_id= -> builds history+forecast from 5m snapshots and signals.
