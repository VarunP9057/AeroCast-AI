
import {
  LayoutDashboard,
  Users,
  Plane,
  Package,
  Activity,
  BrainCircuit,
  FlaskConical,
  Settings,
  RadioTower,
  Radar,
  ChevronLeft,
  ChevronRight,
  Bell
} from 'lucide-react'

const menuItems = [
  { label: 'Overview', icon: LayoutDashboard },
  { label: 'Smart Alerts', icon: Bell },
  { label: 'Passenger Analytics', icon: Users },
  { label: 'Aircraft Analytics', icon: Plane },
  { label: 'Airport Live View', icon: Radar },
  { label: 'Cargo Analytics', icon: Package },
  { label: 'Operational Pressure', icon: Activity },
  { label: 'ANN Forecasting', icon: BrainCircuit },
  { label: 'Model Laboratory', icon: FlaskConical },
]

export default function Sidebar({ activePage, setActivePage, collapsed, onToggle }) {
  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-brand">
        <div className="brand-icon">
          <RadioTower size={25} />
        </div>
        {!collapsed && (
          <div>
            <h2>AEROCAST<span> AI</span></h2>
            <p>INTELLIGENT AIRPORT SYSTEM</p>
          </div>
        )}
      </div>

      <button
        className="sidebar-collapse-button"
        onClick={onToggle}
        type="button"
        aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        {collapsed ? <ChevronRight size={17} /> : <ChevronLeft size={17} />}
      </button>

      {!collapsed && <div className="sidebar-section-label">MAIN MENU</div>}

      <nav className="sidebar-menu">
        {menuItems.map((item) => {
          const Icon = item.icon
          const active = activePage === item.label

          return (
            <button
              key={item.label}
              className={`sidebar-link ${active ? 'active' : ''}`}
              onClick={() => setActivePage(item.label)}
              type="button"
              title={collapsed ? item.label : undefined}
              aria-current={active ? 'page' : undefined}
            >
              <Icon size={19} />
              {!collapsed && <span>{item.label}</span>}
              {!collapsed && active && <span className="active-indicator" />}
            </button>
          )
        })}
      </nav>

      <div className="sidebar-bottom">
        <div className="system-status">
          <div className="status-icon">
            <RadioTower size={20} />
          </div>
          {!collapsed && (
            <div>
              <strong>System Status</strong>
              <p><span className="status-dot" /> All Systems Operational</p>
            </div>
          )}
        </div>

        <button
          className="sidebar-link settings-link"
          type="button"
          title={collapsed ? 'Settings' : undefined}
        >
          <Settings size={19} />
          {!collapsed && <span>Settings</span>}
        </button>

        {!collapsed && (
          <div className="sidebar-footer">
            <div className="footer-line" />
            <p>POWERED BY ARTIFICIAL INTELLIGENCE</p>
            <span>AEROCAST AI · 2026</span>
          </div>
        )}
      </div>
    </aside>
  )
}
