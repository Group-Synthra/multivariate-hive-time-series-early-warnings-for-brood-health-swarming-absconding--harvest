const CLASSIFIER_DERIVED_HUI_URL =
  "/data/harvesting-research/classifier-derived-hui-viva-dashboard.json";

export async function loadClassifierDerivedHuiDashboard() {
  const response = await fetch(CLASSIFIER_DERIVED_HUI_URL, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      "Classifier-derived HUI dashboard data is unavailable. Run " +
        "python scripts/export_classifier_derived_hui_viva_dashboard.py " +
        "from the backend directory.",
    );
  }

  return response.json();
}
