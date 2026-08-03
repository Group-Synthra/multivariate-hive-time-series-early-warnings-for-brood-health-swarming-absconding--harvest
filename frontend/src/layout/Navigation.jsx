import { NAV_ITEMS } from '../config/modules';

export function Navigation({ activePage, onChange }) {
  return (
    <nav className="navigation" aria-label="Main navigation">
      {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
        <button
          key={id}
          type="button"
          className={activePage === id ? 'nav-item active' : 'nav-item'}
          onClick={() => onChange(id)}
        >
          <Icon size={17} />
          <span>{label}</span>
        </button>
      ))}
    </nav>
  );
}
