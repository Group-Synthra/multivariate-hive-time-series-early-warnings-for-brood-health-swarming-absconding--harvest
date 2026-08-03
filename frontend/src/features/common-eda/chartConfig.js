export const SENSOR_OPTIONS = [
  { key: 'temperature', label: 'Temperature', unit: '°C', color: '#dc2626' },
  { key: 'humidity', label: 'Humidity', unit: '%', color: '#0284c7' },
  { key: 'co2', label: 'CO₂', unit: 'ppm', color: '#059669' },
  { key: 'weight', label: 'Hive weight', unit: 'kg', color: '#ca8a04' },
];

export const TARGET_OPTIONS = [
  { key: 'brood_health_healthy_1', label: 'Brood healthy' },
  { key: 'swarming_happened_1', label: 'Swarming' },
  { key: 'absconding_happened_1', label: 'Absconding' },
  { key: 'honey_harvested_1', label: 'Harvesting' },
];

export function sensorOption(key) {
  return SENSOR_OPTIONS.find((item) => item.key === key) || SENSOR_OPTIONS[0];
}

export function targetOption(key) {
  return TARGET_OPTIONS.find((item) => item.key === key) || TARGET_OPTIONS[0];
}
