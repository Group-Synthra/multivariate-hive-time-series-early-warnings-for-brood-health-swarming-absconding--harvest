# Reviewed Dataset Datetime Compatibility Patch

This patch fixes two failures caused by mixing:
- pandas datetime integer values, which may use microseconds, and
- `Timedelta.value`, which uses nanoseconds.

Replace:

`backend/src/multivari/modules/harvesting/reviewed_dataset.py`

with the file in this package.

Then run:

```powershell
ruff check . --fix
ruff check .
pytest -v
```
