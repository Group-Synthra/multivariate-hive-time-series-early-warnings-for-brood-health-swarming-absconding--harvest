import {
  BarChart3,
  HeartPulse,
  Home,
  PackageCheck,
  Waves,
  Wind,
} from 'lucide-react';

export const NAV_ITEMS = [
  { id: 'overview', label: 'Overview', icon: Home },
  { id: 'common-eda', label: 'Common Data EDA', icon: BarChart3 },
  { id: 'brood-health', label: '1. Brood Health', icon: HeartPulse },
  { id: 'swarming', label: '2. Swarming', icon: Waves },
  { id: 'absconding', label: '3. Absconding', icon: Wind },
  { id: 'harvesting', label: '4. Harvesting', icon: PackageCheck },
];

export const MODULES = [
  {
    id: 'brood-health',
    title: 'Brood Health Prediction',
    target: 'brood_health_healthy_1',
    description: 'Current and future brood-health assessment, BHSI and RoD.',
  },
  {
    id: 'swarming',
    title: 'Colony Swarming Prediction',
    target: 'swarming_happened_1',
    description: 'Future swarming-event prediction from multivariate hive patterns.',
  },
  {
    id: 'absconding',
    title: 'Absconding Prediction',
    target: 'absconding_happened_1',
    description: 'Long-term colony-departure risk and deterioration analysis.',
  },
  {
    id: 'harvesting',
    title: 'Honey Harvesting Prediction',
    target: 'honey_harvested_1',
    description: 'Harvest-readiness prediction based on weight and environmental trends.',
  },
];
