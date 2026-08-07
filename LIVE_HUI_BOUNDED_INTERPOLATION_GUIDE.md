# Bounded Application-Level Hourly Interpolation

This patch allows the frozen live-HUI pipeline to run immediately without
modifying PostgreSQL.

## Behavior

- Existing ten-minute readings are aggregated to hourly medians first.
- A fully missing interior hourly run is interpolated only when:
  - it is bounded by observed hours on both sides;
  - required sensors are available at both boundaries;
  - the run does not exceed the configured limit.
- Existing observed values are never overwritten.
- No extrapolation is allowed.
- Longer gaps remain rejected.
- Imputed rows are flagged in memory.
- Evidence confidence is capped at Low while imputed rows remain in the latest
  192-hour model window.
- PostgreSQL is unchanged.

## Automatic transition

As new real readings arrive, the rolling 192-hour model window moves forward.
When the 3 August gap leaves that window, the imputed-hour count becomes zero,
the banner disappears, and inference becomes observed-only automatically.

## Environment

Add to `backend/.env`:

```dotenv
IOT_HOURLY_INTERPOLATION_ENABLED=true
IOT_MAX_INTERPOLATED_GAP_HOURS=8
```

After the viva, disable it with:

```dotenv
IOT_HOURLY_INTERPOLATION_ENABLED=false
```

## Viva wording

> The eight missing hourly buckets were reconstructed only in the application
> feature layer using bounded linear interpolation. The PostgreSQL source
> records were preserved, each imputed hour was flagged, evidence confidence
> was reduced, and the system automatically returns to observed-only inference
> when the reconstructed interval leaves the rolling 192-hour model window.
