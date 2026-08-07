const FINAL_EDA_URL =
  "/data/harvesting-research/final-reviewed-eda-dashboard.json";

export async function loadFinalHarvestEdaDashboard() {
  const response = await fetch(FINAL_EDA_URL, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      `Final EDA dashboard could not be loaded (HTTP ${response.status}).`,
    );
  }

  return response.json();
}
