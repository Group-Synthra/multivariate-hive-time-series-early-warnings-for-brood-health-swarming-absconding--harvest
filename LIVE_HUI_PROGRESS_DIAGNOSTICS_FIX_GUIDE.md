# Live HUI Progress Diagnostic Fix

The current dashboard correctly refuses to display stale HUI predictions, but
the progress bars display `0/168` and `0/192` because the HTTP 422 diagnostic
payload is not being normalized consistently.

This patch:

- reads diagnostics from several possible 422 response layouts;
- derives the hive list from diagnostic records;
- falls back to the first available diagnostic;
- recognizes alternate contiguous-row field names;
- leaves all model logic unchanged.

After applying it, the current value should be approximately `86/168` and
`86/192`, increasing hourly while the newest continuous history remains intact.
