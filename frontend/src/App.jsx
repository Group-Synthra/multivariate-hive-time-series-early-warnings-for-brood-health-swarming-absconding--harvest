import { useState } from 'react';
import { ErrorState } from './components/common/ErrorState';
import { LoadingState } from './components/common/LoadingState';
import { AbscondingPage } from './features/absconding/AbscondingPage';
import { BroodHealthPage } from './features/brood-health/BroodHealthPage';
import { CommonEDAPage } from './features/common-eda/CommonEDAPage';
import { HarvestingPage } from './features/harvesting/HarvestingPage';
import { OverviewPage } from './features/overview/OverviewPage';
import { SwarmingPage } from './features/swarming/SwarmingPage';
import { useEDAData } from './hooks/useEDAData';
import { AppShell } from './layout/AppShell';

function renderPage(activePage, edaData, setActivePage) {
  switch (activePage) {
    case 'common-eda':
      return <CommonEDAPage edaData={edaData} />;
    case 'brood-health':
      return <BroodHealthPage />;
    case 'swarming':
      return <SwarmingPage />;
    case 'absconding':
      return <AbscondingPage />;
    case 'harvesting':
      return <HarvestingPage />;
    default:
      return <OverviewPage edaData={edaData} onOpenModule={setActivePage} />;
  }
}

export default function App() {
  const [activePage, setActivePage] = useState('overview');
  const { edaData, loading, error, refetch } = useEDAData();

  if (loading && !edaData) {
    return <LoadingState />;
  }

  if (error && !edaData) {
    return <ErrorState error={error} onRetry={refetch} />;
  }

  return (
    <AppShell
      activePage={activePage}
      onPageChange={setActivePage}
      onRefresh={refetch}
      refreshing={loading}
    >
      {renderPage(activePage, edaData, setActivePage)}
    </AppShell>
  );
}
