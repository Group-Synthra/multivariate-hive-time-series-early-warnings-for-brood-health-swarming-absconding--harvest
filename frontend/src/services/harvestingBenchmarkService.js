const BENCHMARK_URL =
  "/data/harvesting-research/benchmark-dashboard.json";

export async function loadHarvestingBenchmarkDashboard() {
  const response = await fetch(BENCHMARK_URL, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      "Harvesting benchmark data is unavailable. Run " +
        "python scripts/export_harvesting_benchmark_dashboard.py " +
        "from the backend directory.",
    );
  }

  return response.json();
}
