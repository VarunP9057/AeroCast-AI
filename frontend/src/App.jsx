
import OperationalPressure from './pages/OperationalPressure'
import { useEffect, useState } from 'react'
import AircraftAnalytics from './pages/AircraftAnalytics'
import Sidebar from './components/Sidebar'
import Topbar from './components/Topbar'
import Overview from './pages/Overview'
import PassengerAnalytics from './pages/PassengerAnalytics'
import CargoAnalytics from './pages/CargoAnalytics'
import ANNForecasting from './pages/ANNForecasting'
import ModelLaboratory from './pages/ModelLaboratory'
import AirportLiveView from './pages/AirportLiveView'
import './App.css'
import SmartAlerts from './pages/SmartAlerts'
import { AlertProvider } from './context/AlertContext'

function App() {
  const [activePage, setActivePage] = useState('Overview')

  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    return localStorage.getItem('aerocast-sidebar-collapsed') === 'true'
  })

  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('aerocast-theme') || 'light'
  })

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('aerocast-theme', theme)
  }, [theme])

  useEffect(() => {
    localStorage.setItem(
      'aerocast-sidebar-collapsed',
      String(sidebarCollapsed)
    )
  }, [sidebarCollapsed])

  const toggleSidebar = () => {
    setSidebarCollapsed(current => !current)
  }

  const toggleTheme = () => {
    setTheme(current => current === 'light' ? 'dark' : 'light')
  }

  const renderPage = () => {
    switch (activePage) {
      case 'Overview':
        return (
          <Overview
            onViewAllAlerts={() => setActivePage('Smart Alerts')}
          />
        )

      case 'Smart Alerts':
        return <SmartAlerts />

      case 'Passenger Analytics':
        return <PassengerAnalytics />

      case 'Aircraft Analytics':
        return <AircraftAnalytics />

      case 'Airport Live View':
        return <AirportLiveView />

      case 'Cargo Analytics':
        return <CargoAnalytics />

      case 'Operational Pressure':
        return <OperationalPressure />

      case 'ANN Forecasting':
        return <ANNForecasting />

      case 'Model Laboratory':
        return <ModelLaboratory />

      default:
        return <Overview />
    }
  }

  return (
    <AlertProvider>
      <div
        className={`app-layout ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}
        data-theme={theme}
      >
        <Sidebar
          activePage={activePage}
          setActivePage={setActivePage}
          collapsed={sidebarCollapsed}
          onToggle={toggleSidebar}
        />

        <main className="main-content">
          <Topbar
            theme={theme}
            onToggleTheme={toggleTheme}
            onToggleSidebar={toggleSidebar}
            sidebarCollapsed={sidebarCollapsed}
          />

          <div className="page-container">
            {renderPage()}
          </div>
        </main>
      </div>
    </AlertProvider>
  )
}

export default App
