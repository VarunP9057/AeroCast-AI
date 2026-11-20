import { useMemo, useState } from 'react'
import {
  Bell,
  AlertTriangle,
  CheckCircle,
  Clock3,
  Activity,
  Search,
  Sparkles,
  RotateCcw,
  Check,
} from 'lucide-react'

import { useAlertContext } from '../context/AlertContext'

export default function SmartAlerts() {
  const { alerts, alertStates, setAlertStates } = useAlertContext()

  const [statusFilter, setStatusFilter] = useState('all')
  const [severityFilter, setSeverityFilter] = useState('all')
  const [searchTerm, setSearchTerm] = useState('')

  const updateStatus = (alertId, status) => {
    setAlertStates((previous) => ({
      ...previous,
      [alertId]: status,
    }))
  }

  const getStatus = (alert) =>
    alertStates[alert.id] || 'active'

  const counts = {
    total: alerts.length,
    active: alerts.filter(
      (alert) => getStatus(alert) === 'active'
    ).length,
    acknowledged: alerts.filter(
      (alert) => getStatus(alert) === 'acknowledged'
    ).length,
    resolved: alerts.filter(
      (alert) => getStatus(alert) === 'resolved'
    ).length,
  }

  const filteredAlerts = useMemo(() => {
    return alerts.filter((alert) => {
      const status = alertStates[alert.id] || 'active'

      const matchesStatus =
        statusFilter === 'all' || status === statusFilter

      const matchesSeverity =
        severityFilter === 'all' ||
        alert.severity === severityFilter

      const searchText = searchTerm.toLowerCase()

      const matchesSearch =
        alert.title.toLowerCase().includes(searchText) ||
        alert.detail.toLowerCase().includes(searchText) ||
        alert.recommendation.toLowerCase().includes(searchText)

      return matchesStatus && matchesSeverity && matchesSearch
    })
  }, [alerts, alertStates, statusFilter, severityFilter, searchTerm])

  return (
    <div className="smart-alerts-page">

      <div className="smart-alerts-page-header">
        <div>
          <span className="smart-alerts-eyebrow">
            <Activity size={14} />
            AEROCAST AI MONITORING
          </span>

          <h1>Smart Alerts Management</h1>

          <p>
            Monitor airport activity deviations, review alerts
            and manage operational notifications.
          </p>
        </div>

        <div className="smart-alerts-live-status">
          <span className="smart-alerts-live-dot" />
          Monitoring Active
        </div>
      </div>

      <div className="smart-alert-summary">

        <div className="smart-alert-summary-card">
          <div className="summary-icon total">
            <Bell size={21} />
          </div>
          <div>
            <span>Total Alerts</span>
            <h2>{counts.total}</h2>
            <p>Generated notifications</p>
          </div>
        </div>

        <div className="smart-alert-summary-card">
          <div className="summary-icon active">
            <AlertTriangle size={21} />
          </div>
          <div>
            <span>Active</span>
            <h2>{counts.active}</h2>
            <p>Require attention</p>
          </div>
        </div>

        <div className="smart-alert-summary-card">
          <div className="summary-icon acknowledged">
            <Clock3 size={21} />
          </div>
          <div>
            <span>Acknowledged</span>
            <h2>{counts.acknowledged}</h2>
            <p>Under review</p>
          </div>
        </div>

        <div className="smart-alert-summary-card">
          <div className="summary-icon resolved">
            <CheckCircle size={21} />
          </div>
          <div>
            <span>Resolved</span>
            <h2>{counts.resolved}</h2>
            <p>Completed alerts</p>
          </div>
        </div>

      </div>

      <section className="smart-alert-management-card">

        <div className="smart-alert-management-heading">
          <div>
            <h2>Alert Notifications</h2>
            <p>Review detected airport activity and recommendations.</p>
          </div>

          <span className="smart-alert-total-count">
            {filteredAlerts.length} Results
          </span>
        </div>

        <div className="smart-alert-controls">

          <div className="smart-alert-search">
            <Search size={17} />
            <input
              type="text"
              placeholder="Search alerts..."
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
            />
          </div>

          <select
            value={statusFilter}
            onChange={(event) => setStatusFilter(event.target.value)}
          >
            <option value="all">All Statuses</option>
            <option value="active">Active</option>
            <option value="acknowledged">Acknowledged</option>
            <option value="resolved">Resolved</option>
          </select>

          <select
            value={severityFilter}
            onChange={(event) => setSeverityFilter(event.target.value)}
          >
            <option value="all">All Severities</option>
            <option value="danger">Critical</option>
            <option value="warning">Warning</option>
            <option value="success">Information</option>
          </select>

        </div>

        <div className="smart-alert-management-list">

          {filteredAlerts.length === 0 ? (
            <div className="smart-alert-no-results">
              <Bell size={30} />
              <h3>No alerts found</h3>
              <p>
                No notifications match your selected filters.
              </p>
            </div>
          ) : (
            filteredAlerts.map((alert) => {
              const status = getStatus(alert)

              return (
                <article
                  className={`management-alert-item ${alert.severity}`}
                  key={alert.id}
                >

                  <div className={`management-alert-icon ${alert.severity}`}>
                    {alert.severity === 'success'
                      ? <CheckCircle size={20} />
                      : <AlertTriangle size={20} />}
                  </div>

                  <div className="management-alert-content">

                    <div className="management-alert-title">
                      <h3>{alert.title}</h3>

                      <div className="management-alert-badges">
                        <span className={`severity-badge ${alert.severity}`}>
                          {alert.severity === 'danger'
                            ? 'Critical'
                            : alert.severity === 'warning'
                              ? 'Warning'
                              : 'Information'}
                        </span>

                        <span className={`management-status-badge ${status}`}>
                          {status}
                        </span>
                      </div>
                    </div>

                    <p className="management-alert-detail">
                      {alert.detail}
                    </p>

                    <div className="management-alert-recommendation">
                      <Sparkles size={15} />
                      <span>{alert.recommendation}</span>
                    </div>

                    <div className="management-alert-footer">
                      <span>
                        <Clock3 size={13} />
                        {alert.time}
                      </span>

                      <div className="management-alert-actions">

                        {status === 'active' && (
                          <button
                            onClick={() =>
                              updateStatus(alert.id, 'acknowledged')
                            }
                          >
                            <Check size={14} />
                            Acknowledge
                          </button>
                        )}

                        {status !== 'resolved' && (
                          <button
                            className="resolve-button"
                            onClick={() =>
                              updateStatus(alert.id, 'resolved')
                            }
                          >
                            <CheckCircle size={14} />
                            Resolve
                          </button>
                        )}

                        {status === 'resolved' && (
                          <button
                            onClick={() =>
                              updateStatus(alert.id, 'active')
                            }
                          >
                            <RotateCcw size={14} />
                            Reopen
                          </button>
                        )}

                      </div>
                    </div>

                  </div>
                </article>
              )
            })
          )}

        </div>
      </section>

    </div>
  )
}