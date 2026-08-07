# Live HUI Visual Dashboard Upgrade

This package changes only the Live IoT Prediction presentation.

It keeps the existing:
- PostgreSQL connection
- current HUI calculation
- bounded interpolation
- 24h / 48h / 72h models
- confidence logic
- API routes

The new layout emphasizes:
- Current HUI
- 72-hour forecast
- concise recommendation
- current IoT variables
- compact data-quality indicators

Long research explanations are removed from the main dashboard. Important
limitations remain visible as compact badges.
