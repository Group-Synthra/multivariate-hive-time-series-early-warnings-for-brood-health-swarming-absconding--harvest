const DASHBOARD_URL =
  "/data/harvesting-models/dashboard.json";

export async function loadHarvestingModelDashboard() {
  const response = await fetch(DASHBOARD_URL, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      "Model dashboard data is unavailable. Run " +
        "python scripts/export_harvest_model_results_for_frontend.py " +
        "from the backend directory.",
    );
  }

  return response.json();
}
