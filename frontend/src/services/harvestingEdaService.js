const DATA_ROOT = "/data/harvesting";

async function fetchJson(filename) {
  const response = await fetch(`${DATA_ROOT}/${filename}`, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      `Unable to load ${filename}. Run the harvesting EDA export script first.`,
    );
  }

  return response.json();
}

export async function loadHarvestingEdaDashboard() {
  const [
    summary,
    topFeatures,
    featureComparison,
    sampleCoverage,
  ] = await Promise.all([
    fetchJson("summary.json"),
    fetchJson("top-features.json"),
    fetchJson("feature-comparison.json"),
    fetchJson("sample-coverage.json"),
  ]);

  return {
    summary,
    topFeatures,
    featureComparison,
    sampleCoverage,
  };
}
