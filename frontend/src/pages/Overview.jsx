import heroImage from "../assets/images/aerocast-hero.png";
import { useAlertContext } from '../context/AlertContext'
import { useEffect, useState } from 'react'

import {
  Users,
  Plane,
  Package,
  Activity,
  BrainCircuit,
  Clock3,
  Send,
  Bot,
  User,
  TrendingUp,
  ArrowUpRight,
  MessageCircle,
  Gauge,
  Sparkles,
  Bell,
  AlertTriangle,
  CheckCircle,
  Sun,
  MapPin,
  Box,
  Radio,
  Cloud,
  CloudRain,
  CloudDrizzle,
  CloudSnow,
  CloudLightning,
} from 'lucide-react'

import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  PieChart,
  Pie,
  Cell,
} from 'recharts'

const API_BASE = 'http://127.0.0.1:8000'

const getWeatherInfo = (code) => {
  if (code === 0) return { label: 'Clear Sky', icon: Sun }
  if (code <= 2) return { label: 'Partly Cloudy', icon: Cloud }
  if (code === 3) return { label: 'Overcast', icon: Cloud }
  if (code <= 48) return { label: 'Foggy', icon: Cloud }
  if (code <= 57) return { label: 'Drizzle', icon: CloudDrizzle }
  if (code <= 67) return { label: 'Rain', icon: CloudRain }
  if (code <= 77) return { label: 'Snow', icon: CloudSnow }
  if (code <= 82) return { label: 'Rain Showers', icon: CloudRain }
  if (code <= 86) return { label: 'Snow Showers', icon: CloudSnow }
  if (code <= 99) return { label: 'Thunderstorm', icon: CloudLightning }
  return { label: 'Unknown', icon: Cloud }
}

const formatNumber = (value) =>
  Number(value || 0).toLocaleString('en-US', { maximumFractionDigits: 0 })

const formatMillions = (value) => {
  const number = Number(value || 0)
  if (number >= 1_000_000_000) return `${(number / 1_000_000_000).toFixed(2)}B`
  if (number >= 1_000_000) return `${(number / 1_000_000).toFixed(2)}M`
  if (number >= 1_000) return `${(number / 1_000).toFixed(1)}K`
  return formatNumber(number)
}

const modules = [
  {
    title: 'Passenger Analytics',
    description: 'Passenger volume, airline rankings and terminal traffic.',
    icon: Users,
  },
  {
    title: 'Aircraft Analytics',
    description: 'Aircraft landings, movements and aviation activity.',
    icon: Plane,
  },
  {
    title: 'Cargo Analytics',
    description: 'Airport freight volume and cargo movement trends.',
    icon: Package,
  },
  {
    title: 'Operational Pressure',
    description: 'Integrated airport activity and operational pressure.',
    icon: Activity,
  },
  {
    title: 'ANN Forecasting',
    description: 'AI-powered passenger volume prediction.',
    icon: BrainCircuit,
  },
]

function Overview({ onViewAllAlerts }) {
  const [message, setMessage] = useState('')
  const [messages, setMessages] = useState([
    {
      sender: 'bot',
      text: "Hello! I'm AeroCast AI. Ask me about airport analytics and passenger forecasts.",
    },
  ])
  const [chatOpen, setChatOpen] = useState(false)
  const [alertFilter, setAlertFilter] = useState('all')

  const {
    alerts,
    setAlerts,
    alertStates,
    setAlertStates,
  } = useAlertContext()

  const [analytics, setAnalytics] = useState({
    loading: true,
    error: '',
    passengers: null,
    aircraft: null,
    cargo: null,
    integrated: null,
  })

  const [weather, setWeather] = useState({
    loading: true,
    error: false,
    data: null,
    updatedAt: null,
  })

  useEffect(() => {
    let active = true

    const loadAnalytics = async () => {
      try {
        const [passengerResponse, aircraftResponse, cargoResponse, integratedResponse] =
          await Promise.all([
            fetch(`${API_BASE}/api/analytics/passengers/detailed`),
            fetch(`${API_BASE}/api/analytics/aircraft/monthly`),
            fetch(`${API_BASE}/api/analytics/cargo`),
            fetch(`${API_BASE}/api/analytics/integrated`),
          ])

        if (
          !passengerResponse.ok ||
          !aircraftResponse.ok ||
          !cargoResponse.ok ||
          !integratedResponse.ok
        ) {
          throw new Error('One or more analytics APIs returned an error.')
        }

        const [passengers, aircraft, cargo, integrated] = await Promise.all([
          passengerResponse.json(),
          aircraftResponse.json(),
          cargoResponse.json(),
          integratedResponse.json(),
        ])

        if (active) {
          setAnalytics({
            loading: false,
            error: '',
            passengers,
            aircraft,
            cargo,
            integrated,
          })
        }
      } catch (error) {
        console.error('AeroCast analytics loading error:', error)
        if (active) {
          setAnalytics((previous) => ({
            ...previous,
            loading: false,
            error: error.message || 'Unable to load analytics data.',
          }))
        }
      }
    }

    loadAnalytics()
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    let active = true

    const fetchWeather = async () => {
      try {
        const response = await fetch(
          'https://api.open-meteo.com/v1/forecast?latitude=37.6213&longitude=-122.3790&current=temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m&timezone=America%2FLos_Angeles'
        )
        if (!response.ok) throw new Error('Weather API request failed')
        const result = await response.json()

        if (active) {
          setWeather({
            loading: false,
            error: false,
            data: result.current,
            updatedAt: new Date(),
          })
        }
      } catch (error) {
        console.error('SFO weather error:', error)
        if (active) {
          setWeather((previous) => ({
            ...previous,
            loading: false,
            error: true,
          }))
        }
      }
    }

    fetchWeather()
    const interval = setInterval(fetchWeather, 10 * 60 * 1000)

    return () => {
      active = false
      clearInterval(interval)
    }
  }, [])

  const sendMessage = async () => {
    const userMessage = message.trim()

    if (!userMessage) return

    setMessages((previous) => [
      ...previous,
      { sender: 'user', text: userMessage },
    ])

    setMessage('')

    try {
      const response = await fetch(`${API_BASE}/api/assistant/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Unable to get a response.')
      }

      setMessages((previous) => [
        ...previous,
        { sender: 'bot', text: data.answer },
      ])
    } catch (error) {
      setMessages((previous) => [
        ...previous,
        {
          sender: 'bot',
          text: `Sorry, I couldn't connect to AeroCast AI. ${error.message}`,
        },
      ])
    }
  }


  const passengerData = analytics.passengers?.monthly_traffic || []
  const chartData = passengerData.slice(-12).map((item) => ({
    time: item.month,
    passengers: item.passengers,
  }))

  const latestPassengerMonth =
    analytics.integrated?.latest_month || passengerData.at(-1)?.month || '—'

  const stats = [
    {
      title: 'Total Passengers',
      value: analytics.loading
        ? 'Loading...'
        : analytics.passengers
          ? formatMillions(analytics.passengers.total_passengers)
          : 'Unavailable',
      subtitle: 'Historical passenger total',
      icon: Users,
      color: 'green',
    },
    {
      title: 'Aircraft Movements',
      value: analytics.loading
        ? 'Loading...'
        : analytics.aircraft
          ? formatNumber(analytics.aircraft.total_landings)
          : 'Unavailable',
      subtitle: 'Recorded aircraft landings',
      icon: Plane,
      color: 'yellow',
    },
    {
      title: 'Cargo Handled',
      value: analytics.loading
        ? 'Loading...'
        : analytics.cargo
          ? `${(analytics.cargo.total_cargo_weight_lbs / 1_000_000_000).toFixed(2)}B lbs`
          : 'Unavailable',
      subtitle: 'Historical cargo weight',
      icon: Package,
      color: 'orange',
    },
    {
      title: 'Operational Pressure',
      value: analytics.loading
        ? 'Loading...'
        : analytics.integrated
          ? 'Data Connected'
          : 'Unavailable',
      subtitle: analytics.integrated
        ? `Latest month: ${analytics.integrated.latest_month}`
        : 'Integrated airport activity',
      icon: Gauge,
      color: 'green',
    },
  ]

  const performanceData = analytics.integrated
    ? [
      { name: 'Passengers', value: 68 },
      { name: 'Aircraft', value: 20 },
      { name: 'Cargo', value: 12 },
    ]
    : []

  const weatherInfo = weather.data
    ? getWeatherInfo(weather.data.weather_code)
    : { label: 'Loading weather...', icon: Cloud }
  const WeatherIcon = weatherInfo.icon

  const generatedAlerts = []

  const passengerHistory = analytics.passengers?.monthly_traffic || []
  const aircraftHistory = analytics.aircraft?.monthly_traffic || []

  const latestPassenger = passengerHistory.at(-1)
  const previousPassenger = passengerHistory.at(-2)

  const latestAircraft = aircraftHistory.at(-1)
  const previousAircraft = aircraftHistory.at(-2)

  const getPercentageChange = (current, previous) => {
    if (!previous || previous <= 0) return null
    return ((current - previous) / previous) * 100
  }

  const addTrafficAlert = (id, name, change, month) => {
    if (change === null || Math.abs(change) < 20) return

    const increase = change > 0
    const severity = increase && change >= 35 ? 'danger' : 'warning'

    generatedAlerts.push({
      id,
      severity,
      title: `${name} ${increase ? 'spike' : 'drop'} detected`,
      detail: `${Math.abs(change).toFixed(1)}% ${increase ? 'increase' : 'decrease'} compared with the previous month.`,
      recommendation: increase
        ? 'Review staffing, capacity and resource allocation.'
        : 'Review traffic trends and verify the latest data.',
      time: month || 'Latest available month',
    })
  }

  if (latestPassenger && previousPassenger) {
    addTrafficAlert(
      'passenger-change',
      'Passenger traffic',
      getPercentageChange(
        Number(latestPassenger.passengers),
        Number(previousPassenger.passengers)
      ),
      latestPassenger.month
    )
  }

  if (latestAircraft && previousAircraft) {
    addTrafficAlert(
      'aircraft-change',
      'Aircraft movements',
      getPercentageChange(
        Number(latestAircraft.landings),
        Number(previousAircraft.landings)
      ),
      latestAircraft.month
    )
  }

  const cargoLatest = Number(
    analytics.cargo?.latest_cargo_weight_lbs || 0
  )

  const cargoAverage = Number(
    analytics.cargo?.average_monthly_cargo_lbs || 0
  )

  const cargoRatio =
    cargoAverage > 0 ? cargoLatest / cargoAverage : 0

  if (cargoRatio >= 1.2) {
    generatedAlerts.push({
      id: 'cargo-elevation',
      severity: cargoRatio >= 1.35 ? 'danger' : 'warning',
      title: 'Elevated cargo volume',
      detail: `Latest cargo volume is ${((cargoRatio - 1) * 100).toFixed(1)}% above its historical monthly average.`,
      recommendation: 'Review freight handling capacity and cargo resources.',
      time: analytics.cargo?.latest_month || 'Latest available month',
    })
  }

  const passengerAverage = passengerHistory.length
    ? passengerHistory.reduce(
      (sum, item) => sum + Number(item.passengers || 0),
      0
    ) / passengerHistory.length
    : 0

  const aircraftAverage = aircraftHistory.length
    ? aircraftHistory.reduce(
      (sum, item) => sum + Number(item.landings || 0),
      0
    ) / aircraftHistory.length
    : 0

  const elevatedIndicators = [
    passengerAverage > 0 &&
    Number(latestPassenger?.passengers || 0) >= passengerAverage * 1.2,
    aircraftAverage > 0 &&
    Number(latestAircraft?.landings || 0) >= aircraftAverage * 1.2,
    cargoRatio >= 1.2,
  ].filter(Boolean).length

  if (elevatedIndicators >= 2) {
    generatedAlerts.push({
      id: 'operational-pressure',
      severity: elevatedIndicators === 3 ? 'danger' : 'warning',
      title: 'Operational pressure detected',
      detail: `${elevatedIndicators} airport activity indicators are above their historical averages.`,
      recommendation: 'Review terminal capacity, ground operations and resource allocation.',
      time: latestPassenger?.month || latestAircraft?.month || 'Latest available month',
    })
  }

  if (
    !analytics.loading &&
    !analytics.error &&
    generatedAlerts.length === 0
  ) {
    generatedAlerts.push({
      id: 'system-healthy',
      severity: 'success',
      title: 'Airport activity monitoring active',
      detail: 'No significant traffic deviations detected by the current monitoring rules.',
      recommendation: 'Continue monitoring passenger, aircraft and cargo activity.',
      time: latestPassenger?.month || 'Latest available month',
    })
  }
  useEffect(() => {
    setAlerts(generatedAlerts)
  }, [analytics, setAlerts])

  const visibleAlerts = alerts.filter((alert) => {
    const status = alertStates[alert.id] || 'active'

    if (alertFilter === 'all') return true

    return status === alertFilter
  })

  const alertCounts = {
    all: alerts.length,
    active: alerts.filter(
      (alert) => (alertStates[alert.id] || 'active') === 'active'
    ).length,
    acknowledged: alerts.filter(
      (alert) => alertStates[alert.id] === 'acknowledged'
    ).length,
    resolved: alerts.filter(
      (alert) => alertStates[alert.id] === 'resolved'
    ).length,
  }



  return (
    <div className="aero-dashboard reference-dashboard">
      <section
        className="airport-hero reference-hero"
        style={{
          backgroundImage: `linear-gradient(90deg, rgba(5, 15, 32, 0.78), rgba(5, 15, 32, 0.10)), url(${heroImage})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center 75%',
          backgroundRepeat: 'no-repeat'
        }}

      >
        <div className="hero-content">
          <span className="hero-tag">
            <Sparkles size={14} />
            SFO · SAN FRANCISCO, USA
          </span>
          <h1>
            AeroCast <span>AI</span>
          </h1>
          <h2>
            ANN-Based Airport Passenger Volume
            <br />
            Prediction and Traffic Pattern Analysis
          </h2>
          <button className="hero-button">
            Explore Analytics <ArrowUpRight size={17} />
          </button>
        </div>

        <div className="weather-card">
          <div className="weather-main">
            <WeatherIcon size={40} />
            <div>
              <strong>
                {weather.data
                  ? `${Math.round(weather.data.temperature_2m)}°C`
                  : '--°C'}
              </strong>
              <span>
                {weather.error ? 'Weather unavailable' : weatherInfo.label}
              </span>
            </div>
          </div>
          <div className="weather-details">
            <span>
              Feels like:{' '}
              {weather.data
                ? `${Math.round(weather.data.apparent_temperature)}°C`
                : '--'}
            </span>
            <span>
              Humidity:{' '}
              {weather.data ? `${weather.data.relative_humidity_2m}%` : '--'}
            </span>
            <span>
              Wind:{' '}
              {weather.data ? `${weather.data.wind_speed_10m} km/h` : '--'}
            </span>
          </div>
          <div className="weather-location">
            <MapPin size={14} />
            SFO, California
          </div>
          <div className="weather-updated">
            {weather.updatedAt
              ? `Updated ${weather.updatedAt.toLocaleTimeString('en-US', {
                timeZone: 'America/Los_Angeles',
                hour: '2-digit',
                minute: '2-digit',
              })} PT`
              : 'Connecting to live weather...'}
          </div>
        </div>

        <div className="hero-bottom-label">AIRPORT INTELLIGENCE PLATFORM</div>
      </section>

      <section className="dashboard-stats">
        {stats.map((stat) => {
          const Icon = stat.icon
          return (
            <article className="dashboard-stat-card reference-stat" key={stat.title}>
              <div className={`dashboard-stat-icon ${stat.color}`}>
                <Icon size={27} />
              </div>
              <div className="dashboard-stat-content">
                <span>{stat.title}</span>
                <h2>{stat.value}</h2>
                <p className="stat-growth">
                  {stat.title === 'Operational Pressure' ? (
                    <Activity size={14} />
                  ) : (
                    <TrendingUp size={14} />
                  )}
                  {stat.subtitle}
                </p>
              </div>
              <TrendingUp className="stat-trend-icon" size={17} />
            </article>
          )
        })}
      </section>

      {analytics.error && (
        <p className="analytics-error">
          Analytics connection issue: {analytics.error}. Check that the FastAPI
          backend is running on port 8000.
        </p>
      )}

      <section className="reference-analytics-grid">
        <article className="analytics-panel flight-panel">
          <div className="panel-heading">
            <div>
              <span className="panel-icon"><Activity size={19} /></span>
              <h3>Passenger Trend</h3>
            </div>
            <span className="panel-period">Latest 12 Months</span>
          </div>
          <div className="flight-chart">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="flightGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#00D9FF" stopOpacity={0.38} />
                    <stop offset="95%" stopColor="#00D9FF" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#dce4ed" strokeDasharray="4 4" />
                <XAxis
                  dataKey="time"
                  stroke="#718096"
                  tick={{ fontSize: 10 }}
                  tickFormatter={(value) => value.slice(2)}
                />
                <YAxis
                  stroke="#718096"
                  tick={{ fontSize: 11 }}
                  tickFormatter={(value) => `${(value / 1_000_000).toFixed(1)}M`}
                />
                <Tooltip
                  formatter={(value) => [formatNumber(value), 'Passengers']}
                  contentStyle={{
                    background: '#ffffff',
                    border: '1px solid #dce4ed',
                    borderRadius: '9px',
                    color: '#172235',
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="passengers"
                  stroke="#00D9FF"
                  strokeWidth={3}
                  fill="url(#flightGradient)"
                  dot={{ r: 3, fill: '#ffffff', stroke: '#647BFF', strokeWidth: 2 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <p className="data-source-note">
            Source: SFO passenger analytics API · Latest data: {latestPassengerMonth}
          </p>
        </article>

        <article className="analytics-panel performance-panel">
          <div className="panel-heading">
            <div>
              <span className="panel-icon"><Box size={19} /></span>
              <h3>Airport Activity Overview</h3>
            </div>
          </div>
          <div className="activity-overview-content">
            <div className="performance-chart">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={performanceData}
                    dataKey="value"
                    nameKey="name"
                    innerRadius="65%"
                    outerRadius="88%"
                    paddingAngle={3}
                    stroke="none"
                  >
                    <Cell fill="#00D9FF" />
                    <Cell fill="#647BFF" />
                    <Cell fill="#A78BFA" />
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
              <div className="performance-center">
                <strong>{analytics.integrated ? 'SFO' : '--'}</strong>
                <span>Activity</span>
              </div>
            </div>
            <div className="performance-legend">
              <span><i className="legend-lime" /> Passengers <b>{analytics.integrated ? 'Connected' : '—'}</b></span>
              <span><i className="legend-yellow" /> Aircraft <b>{analytics.integrated ? 'Connected' : '—'}</b></span>
              <span><i className="legend-grey" /> Cargo <b>{analytics.integrated ? 'Connected' : '—'}</b></span>
            </div>
          </div>
          <p className="data-source-note">
            Data availability indicators; not percentage shares.
          </p>
        </article>

        <article className="live-view-card">
          <div className="live-view-heading">
            <span><i /> Airport Live View</span>
            <button onClick={() => window.alert('3D Twin integration will be added in a later phase.')}>
              View 3D Twin <ArrowUpRight size={14} />
            </button>
          </div>
          <div className="live-view-image">
            <img
              src="/images/twins.png"
              alt="AeroCast AI 3D Airport Digital Twin"
            />
          </div>
          <div className="live-view-caption">
            <Radio size={12} /> SFO AIRPORT · VISUAL PREVIEW
          </div>
        </article>
      </section>

      <section className="reference-bottom-grid">
        <article className="activity-list-card">
          <div className="bottom-card-heading">
            <div>
              <Clock3 size={21} />
              <h3>Recent Activity</h3>
            </div>
            <span>View All <ArrowUpRight size={14} /></span>
          </div>
          <div className="activity-list">
            <div className="activity-list-item">
              <i className="activity-dot green" />
              <p>Passenger analytics data connected</p>
              <time>Live API</time>
            </div>
            <div className="activity-list-item">
              <i className="activity-dot green" />
              <p>Aircraft analytics data connected</p>
              <time>Live API</time>
            </div>
            <div className="activity-list-item">
              <i className="activity-dot green" />
              <p>Cargo analytics data connected</p>
              <time>Live API</time>
            </div>
          </div>
        </article>

        <article className="activity-list-card alerts-card">
          <div className="bottom-card-heading">
            <div>
              <Bell size={21} />
              <h3>Smart Alerts</h3>
            </div>
            <button
              type="button"
              className="alerts-view-all"
              onClick={onViewAllAlerts}
            >
              View All <ArrowUpRight size={14} />
            </button>
          </div>
          <div className="activity-list">

            <div className="smart-alerts-content">
              <div className="alert-filter-bar">
                {['all', 'active', 'acknowledged', 'resolved'].map((filter) => (
                  <button
                    key={filter}
                    className={alertFilter === filter ? 'selected' : ''}
                    onClick={() => setAlertFilter(filter)}
                  >
                    {filter.charAt(0).toUpperCase() + filter.slice(1)}
                    <span>{alertCounts[filter]}</span>
                  </button>
                ))}
              </div>

              <div className="smart-alert-list">
                {analytics.loading ? (
                  <p className="alert-empty">Analyzing airport activity...</p>
                ) : analytics.error ? (
                  <p className="alert-empty">Unable to generate alerts. Check analytics API.</p>
                ) : visibleAlerts.length === 0 ? (
                  <p className="alert-empty">No alerts in this category.</p>
                ) : (
                  visibleAlerts.map((alert) => {
                    const status = alertStates[alert.id] || 'active'

                    return (
                      <div className={`smart-alert-item ${alert.severity}`} key={alert.id}>
                        <div className={`alert-icon ${alert.severity}`}>
                          {alert.severity === 'success'
                            ? <CheckCircle size={15} />
                            : <AlertTriangle size={15} />}
                        </div>

                        <div className="smart-alert-info">
                          <div className="smart-alert-title-row">
                            <strong>{alert.title}</strong>
                            <span className={`alert-status ${status}`}>
                              {status}
                            </span>
                          </div>

                          <p>{alert.detail}</p>

                          <div className="alert-recommendation">
                            <Sparkles size={12} />
                            {alert.recommendation}
                          </div>

                          <time>{alert.time}</time>

                          {status !== 'resolved' && (
                            <div className="alert-actions">
                              {status === 'active' && (
                                <button
                                  onClick={() =>
                                    setAlertStates((previous) => ({
                                      ...previous,
                                      [alert.id]: 'acknowledged',
                                    }))
                                  }
                                >
                                  Acknowledge
                                </button>
                              )}

                              <button
                                onClick={() =>
                                  setAlertStates((previous) => ({
                                    ...previous,
                                    [alert.id]: 'resolved',
                                  }))
                                }
                              >
                                Resolve
                              </button>
                            </div>
                          )}

                          {status === 'resolved' && (
                            <button
                              className="alert-reopen"
                              onClick={() =>
                                setAlertStates((previous) => ({
                                  ...previous,
                                  [alert.id]: 'active',
                                }))
                              }
                            >
                              Reopen alert
                            </button>
                          )}
                        </div>
                      </div>
                    )
                  })
                )}
              </div>
            </div>

            <div className="activity-list-item">
              <span className="alert-icon success"><CheckCircle size={15} /></span>
              <p>Aircraft API connection verified</p>
              <time>API status</time>
            </div>
            <div className="activity-list-item">
              <span className="alert-icon success"><CheckCircle size={15} /></span>
              <p>Cargo API connection verified</p>
              <time>API status</time>
            </div>
          </div>
        </article>

        <article className="ai-assistant-banner">
          <div className="ai-assistant-content">
            <span>AI</span>
            <h2>Airport Assistant</h2>
            <p>
              Ask anything about airport analytics,
              forecasts and more...
            </p>
            <button onClick={() => setChatOpen(true)}>
              Chat Now <ArrowUpRight size={16} />
            </button>
          </div>
          <div className="robot-image" />
        </article>
      </section>

      <section className="dashboard-modules-section">
        <div className="dashboard-section-title">
          <div>
            <span>EXPLORE AEROCAST AI</span>
            <h2>Key Analytics Modules</h2>
          </div>
          <span>05 MODULES</span>
        </div>
        <div className="dashboard-module-grid">
          {modules.map((module) => {
            const Icon = module.icon
            return (
              <article className="dashboard-module-card" key={module.title}>
                <div className="dashboard-module-icon">
                  <Icon size={24} />
                </div>
                <h3>{module.title}</h3>
                <p>{module.description}</p>
                <span className="module-explore">
                  Explore <ArrowUpRight size={15} />
                </span>
              </article>
            )
          })}
        </div>
      </section>

      {chatOpen && (
        <div className="assistant-chat-window">
          <div className="assistant-chat-heading">
            <div>
              <Bot size={19} />
              <strong>AeroCast AI Assistant</strong>
            </div>
            <button onClick={() => setChatOpen(false)}>×</button>
          </div>
          <div className="assistant-chat-messages">
            {messages.map((item, index) => (
              <div className={`chat-message ${item.sender}`} key={index}>
                <div className="chat-message-avatar">
                  {item.sender === 'bot' ? <Bot size={17} /> : <User size={17} />}
                </div>
                <div className="chat-message-text">{item.text}</div>
              </div>
            ))}
          </div>
          <div className="assistant-chat-input">
            <input
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') sendMessage()
              }}
              placeholder="Ask AeroCast AI..."
            />
            <button onClick={sendMessage} aria-label="Send message">
              <Send size={17} />
            </button>
          </div>
          <p className="assistant-chat-note">
            <MessageCircle size={12} />
            Powered by AeroCast AI Analytics
          </p>
        </div>
      )}
    </div>
  )
}

export default Overview
