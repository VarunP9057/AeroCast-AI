import { Search, Bell, Plane, Menu, Sun, Moon } from 'lucide-react'

function Topbar({ theme, onToggleTheme, onToggleSidebar, sidebarCollapsed }) {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <button
          className="topbar-icon-button sidebar-menu-toggle"
          onClick={onToggleSidebar}
          type="button"
          aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          <Menu size={19} />
        </button>
        <div>
          <div className="breadcrumb">
            <span>Workspace</span>
            <span>/</span>
            <strong>AeroCast AI</strong>
          </div>
          <p>Airport Intelligence &amp; Analytics Platform</p>
        </div>
      </div>

      <div className="topbar-right">
        <div className="location-label">
          <Plane size={16} />
          <span>San Francisco · SFO</span>
        </div>

        <div className="search-box">
          <Search size={17} />
          <input type="text" placeholder="Search analytics..." />
        </div>

        <button
          className="topbar-icon-button theme-toggle-button"
          onClick={onToggleTheme}
          type="button"
          aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}
          title={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}
        >
          {theme === 'light' ? <Moon size={18} /> : <Sun size={18} />}
        </button>

        <button className="notification-button" type="button" aria-label="Notifications">
          <Bell size={19} />
        </button>

        <div className="profile-avatar">VP</div>
      </div>
    </header>
  )
}

export default Topbar
