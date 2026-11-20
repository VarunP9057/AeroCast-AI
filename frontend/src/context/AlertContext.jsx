
import { createContext, useContext, useState } from 'react'

const AlertContext = createContext(null)

export function AlertProvider({ children }) {
  const [alerts, setAlerts] = useState([])
  const [alertStates, setAlertStates] = useState({})

  const updateAlertStatus = (alertId, status) => {
    setAlertStates((previous) => ({
      ...previous,
      [alertId]: status,
    }))
  }

  return (
    <AlertContext.Provider
      value={{
        alerts,
        setAlerts,
        alertStates,
        setAlertStates,
        updateAlertStatus,
      }}
    >
      {children}
    </AlertContext.Provider>
  )
}

export function useAlertContext() {
  const context = useContext(AlertContext)

  if (!context) {
    throw new Error(
      'useAlertContext must be used inside AlertProvider'
    )
  }

  return context
}
