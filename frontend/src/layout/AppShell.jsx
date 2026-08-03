import { Header } from './Header';
import { Navigation } from './Navigation';

export function AppShell({ activePage, onPageChange, onRefresh, refreshing, children }) {
  return (
    <div className="app-shell">
      <Header onRefresh={onRefresh} refreshing={refreshing} />
      <Navigation activePage={activePage} onChange={onPageChange} />
      <main className="page-content">{children}</main>
    </div>
  );
}
