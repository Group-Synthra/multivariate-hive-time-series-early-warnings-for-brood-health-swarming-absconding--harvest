const PROVISIONAL_HUI_URL =
  "/data/harvesting-research/provisional-hui-dashboard.json";

export async function loadProvisionalHuiDashboard() {
  const response = await fetch(PROVISIONAL_HUI_URL, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      "Provisional HUI data is unavailable. Run " +
        "python scripts/export_provisional_hui_dashboard.py " +
        "from the backend directory.",
    );
  }

  return response.json();
}
