# Polished Live HUI Dashboard

This package keeps **HUI Decision Support** unchanged and upgrades the
separate **Live IoT Prediction** tab.

It adds:

- a current-HUI gauge;
- 24h, 48h and 72h future-HUI cards;
- recommendation and evidence-confidence panels;
- current internal/external IoT variables;
- battery and freshness information;
- PostgreSQL sensor snapshots even while HUI history is incomplete;
- history-progress bars;
- domain-shift warnings.

The screen never invents HUI values. Current variables appear immediately.
Predictions appear automatically after the newest continuous history satisfies
the frozen model requirements.
