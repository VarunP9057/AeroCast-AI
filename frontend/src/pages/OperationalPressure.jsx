
import { useEffect, useMemo, useState } from 'react'
import axios from 'axios'

import {
  Activity,
  Users,
  Plane,
  Gauge,
  TrendingUp,
  TrendingDown,
  CalendarDays,
  AlertTriangle,
  RefreshCw
} from 'lucide-react'

import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from 'recharts'

import './OperationalPressure.css'

const API = 'http://127.0.0.1:8000/api/analytics'

const numberFormat = value =>
  new Intl.NumberFormat('en-US').format(Math.round(value || 0))

const compactFormat = value => {
  if (value >= 1000000) return `${(value / 1000000).toFixed(2)}M`
  if (value >= 1000) return `${(value / 1000).toFixed(1)}K`
  return numberFormat(value)
}

function OperationalPressure() {
  const [passengerData, setPassengerData] = useState([])
  const [aircraftData, setAircraftData] = useState([])
  const [passengerSummary, setPassengerSummary] = useState(null)
  const [aircraftSummary, setAircraftSummary] = useState(null)
  const [period, setPeriod] = useState('ALL')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true)
        setError('')

        const [passengers, aircraft, passengerMonthly, aircraftMonthly] =
          await Promise.all([
            axios.get(`${API}/passengers`),
            axios.get(`${API}/aircraft`),
            axios.get(`${API}/passengers/detailed`),
            axios.get(`${API}/aircraft/monthly`)
          ])

        setPassengerSummary(passengers.data)
        setAircraftSummary(aircraft.data)

        setPassengerData(passengerMonthly.data.monthly_traffic || [])
        setAircraftData(aircraftMonthly.data.monthly_traffic || [])
      } catch (err) {
        console.error('Operational analytics error:', err)
        setError('Unable to load operational data. Please check the backend API.')
      } finally {
        setLoading(false)
      }
    }

    loadData()
  }, [])

  const mergedData = useMemo(() => {
    const aircraftMap = new Map(
      aircraftData.map(item => [item.month, item.landings])
    )

    return passengerData
      .filter(item => aircraftMap.has(item.month))
      .map(item => {
        const passengers = Number(item.passengers || 0)
        const landings = Number(aircraftMap.get(item.month) || 0)

        return {
          month: item.month,
          passengers,
          landings
        }
      })
      .sort((a, b) => a.month.localeCompare(b.month))
  }, [passengerData, aircraftData])

  const filteredData = useMemo(() => {
    if (period === 'ALL') return mergedData

    const years = Number(period)
    const latestDate = mergedData[mergedData.length - 1]?.month

    if (!latestDate) return []

    const latestYear = Number(latestDate.slice(0, 4))
    const startYear = latestYear - years + 1

    return mergedData.filter(
      item => Number(item.month.slice(0, 4)) >= startYear
    )
  }, [mergedData, period])

  const chartData = useMemo(() => {
    const maxPassengers = Math.max(
      ...mergedData.map(item => item.passengers),
      1
    )

    const maxLandings = Math.max(
      ...mergedData.map(item => item.landings),
      1
    )

    return filteredData.map(item => {
      const passengerIndex = (item.passengers / maxPassengers) * 100
      const aircraftIndex = (item.landings / maxLandings) * 100

      return {
        ...item,
        passengerIndex,
        aircraftIndex,
        pressureIndex: (passengerIndex + aircraftIndex) / 2
      }
    })
  }, [filteredData, mergedData])

  const latest = chartData[chartData.length - 1]

  const peakPressure = useMemo(() => {
    if (!chartData.length) return null

    return chartData.reduce((highest, item) =>
      item.pressureIndex > highest.pressureIndex ? item : highest
    )
  }, [chartData])

  const lowestPressure = useMemo(() => {
    if (!chartData.length) return null

    return chartData.reduce((lowest, item) =>
      item.pressureIndex < lowest.pressureIndex ? item : lowest
    )
  }, [chartData])

  const pressureLabel = value => {
    if (value >= 85) return 'Very High'
    if (value >= 65) return 'High'
    if (value >= 40) return 'Moderate'
    return 'Low'
  }

  const pressureClass = value => {
    if (value >= 85) return 'critical'
    if (value >= 65) return 'high'
    if (value >= 40) return 'moderate'
    return 'low'
  }

  const formatMonth = month => {
    if (!month) return 'N/A'

    const [year, monthNumber] = month.split('-')
    return new Date(Number(year), Number(monthNumber) - 1)
      .toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
  }

  const averagePassengers = filteredData.length
    ? filteredData.reduce((sum, item) => sum + item.passengers, 0) /
      filteredData.length
    : 0

  const averageLandings = filteredData.length
    ? filteredData.reduce((sum, item) => sum + item.landings, 0) /
      filteredData.length
    : 0

  if (loading) {
    return (
      <div className="pressure-page pressure-loading">
        <RefreshCw className="pressure-spinner" size={30} />
        <p>Loading operational pressure analytics...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="pressure-page">
        <div className="pressure-error">
          <AlertTriangle size={25} />
          <h3>Operational Data Unavailable</h3>
          <p>{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="pressure-page">

      <div className="pressure-header">
        <div>
          <div className="pressure-eyebrow">
            <Activity size={15} />
            AIRPORT OPERATIONS INTELLIGENCE
          </div>

          <h1>Operational Pressure Analysis</h1>

          <p>
            Monitor passenger activity, aircraft movements and
            historical operational pressure at San Francisco International Airport.
          </p>
        </div>

        <div className="pressure-live-status">
          <span className="pressure-status-dot"></span>
          DATA CONNECTED
        </div>
      </div>

      <div className="pressure-filter-row">
        <div className="pressure-filter-label">
          <CalendarDays size={17} />
          Analysis Period
        </div>

        <div className="pressure-filters">
          {[
            { label: 'All Time', value: 'ALL' },
            { label: '10 Years', value: '10' },
            { label: '5 Years', value: '5' },
            { label: '3 Years', value: '3' },
            { label: '1 Year', value: '1' }
          ].map(option => (
            <button
              key={option.value}
              className={period === option.value ? 'active' : ''}
              onClick={() => setPeriod(option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      <div className="pressure-kpi-grid">

        <div className="pressure-kpi-card">
          <div className="pressure-kpi-top">
            <div className="pressure-kpi-icon passengers">
              <Users size={21} />
            </div>
            <span className="pressure-kpi-caption">PASSENGER ACTIVITY</span>
          </div>

          <h2>{compactFormat(passengerSummary?.total_passengers)}</h2>
          <p>Total passengers recorded</p>

          <div className="pressure-kpi-footer">
            <TrendingUp size={15} />
            Historical dataset
          </div>
        </div>

        <div className="pressure-kpi-card">
          <div className="pressure-kpi-top">
            <div className="pressure-kpi-icon aircraft">
              <Plane size={21} />
            </div>
            <span className="pressure-kpi-caption">AIRCRAFT MOVEMENTS</span>
          </div>

          <h2>{compactFormat(aircraftSummary?.total_landings)}</h2>
          <p>Total aircraft landings</p>

          <div className="pressure-kpi-footer">
            <TrendingUp size={15} />
            Historical dataset
          </div>
        </div>

        <div className="pressure-kpi-card">
          <div className="pressure-kpi-top">
            <div className="pressure-kpi-icon activity">
              <Activity size={21} />
            </div>
            <span className="pressure-kpi-caption">LATEST PRESSURE INDEX</span>
          </div>

          <h2>{latest ? `${latest.pressureIndex.toFixed(1)}%` : 'N/A'}</h2>
          <p>
            {latest ? pressureLabel(latest.pressureIndex) : 'No data available'}
          </p>

          <div className={`pressure-kpi-footer ${pressureClass(latest?.pressureIndex || 0)}`}>
            <Gauge size={15} />
            Relative historical activity
          </div>
        </div>

        <div className="pressure-kpi-card">
          <div className="pressure-kpi-top">
            <div className="pressure-kpi-icon peak">
              <TrendingUp size={21} />
            </div>
            <span className="pressure-kpi-caption">PEAK PRESSURE PERIOD</span>
          </div>

          <h2>{peakPressure ? peakPressure.pressureIndex.toFixed(1) + '%' : 'N/A'}</h2>
          <p>{peakPressure ? formatMonth(peakPressure.month) : 'No data available'}</p>

          <div className="pressure-kpi-footer">
            <CalendarDays size={15} />
            Highest relative activity
          </div>
        </div>

      </div>

      <div className="pressure-section-heading">
        <div>
          <h2>Airport Traffic Monitoring</h2>
          <p>Passenger and aircraft activity over the selected period</p>
        </div>
      </div>

      <div className="pressure-chart-card">
        <div className="pressure-chart-heading">
          <div>
            <h3>Passenger vs Aircraft Movement</h3>
            <p>Monthly comparison using separate measurement axes</p>
          </div>
          <span className="pressure-chart-tag">MONTHLY DATA</span>
        </div>

        <div className="pressure-chart">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData}>
              <defs>
                <linearGradient id="passengerGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00D9FF" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#00D9FF" stopOpacity={0} />
                </linearGradient>
              </defs>

              <CartesianGrid strokeDasharray="3 3" vertical={false} />

              <XAxis
                dataKey="month"
                tickFormatter={value => value.slice(0, 4)}
                minTickGap={35}
              />

              <YAxis
                yAxisId="passengers"
                tickFormatter={compactFormat}
                width={65}
              />

              <YAxis
                yAxisId="landings"
                orientation="right"
                tickFormatter={compactFormat}
                width={65}
              />

              <Tooltip
                labelFormatter={formatMonth}
                formatter={(value, name) => [
                  numberFormat(value),
                  name
                ]}
              />

              <Legend />

              <Area
                yAxisId="passengers"
                type="monotone"
                dataKey="passengers"
                name="Passengers"
                stroke="#00BFEA"
                fill="url(#passengerGradient)"
                strokeWidth={2}
              />

              <Line
                yAxisId="landings"
                type="monotone"
                dataKey="landings"
                name="Aircraft Landings"
                stroke="#8B7CFF"
                strokeWidth={2}
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="pressure-section-heading">
        <div>
          <h2>Relative Operational Pressure</h2>
          <p>Normalized activity indicators based on historical maximum values</p>
        </div>
      </div>

      <div className="pressure-chart-grid">

        <div className="pressure-chart-card">
          <div className="pressure-chart-heading">
            <div>
              <h3>Operational Activity Index</h3>
              <p>Passenger and aircraft activity normalized to 100%</p>
            </div>
          </div>

          <div className="pressure-chart pressure-small-chart">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />

                <XAxis
                  dataKey="month"
                  tickFormatter={value => value.slice(0, 4)}
                  minTickGap={30}
                />

                <YAxis domain={[0, 100]} unit="%" />

                <Tooltip
                  labelFormatter={formatMonth}
                  formatter={value => `${Number(value).toFixed(1)}%`}
                />

                <Legend />

                <Line
                  type="monotone"
                  dataKey="passengerIndex"
                  name="Passenger Index"
                  stroke="#00BFEA"
                  strokeWidth={2}
                  dot={false}
                />

                <Line
                  type="monotone"
                  dataKey="aircraftIndex"
                  name="Aircraft Index"
                  stroke="#8B7CFF"
                  strokeWidth={2}
                  dot={false}
                />

                <Line
                  type="monotone"
                  dataKey="pressureIndex"
                  name="Combined Activity Index"
                  stroke="#84CC16"
                  strokeWidth={3}
                  dot={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="pressure-chart-card">
          <div className="pressure-chart-heading">
            <div>
              <h3>Monthly Average Activity</h3>
              <p>Average passenger and aircraft volume for this period</p>
            </div>
          </div>

          <div className="pressure-average-list">
            <div className="pressure-average-item">
              <div className="pressure-average-icon passengers">
                <Users size={20} />
              </div>
              <div>
                <span>Average monthly passengers</span>
                <strong>{numberFormat(averagePassengers)}</strong>
              </div>
            </div>

            <div className="pressure-average-item">
              <div className="pressure-average-icon aircraft">
                <Plane size={20} />
              </div>
              <div>
                <span>Average monthly landings</span>
                <strong>{numberFormat(averageLandings)}</strong>
              </div>
            </div>

            <div className="pressure-average-item">
              <div className="pressure-average-icon activity">
                <Gauge size={20} />
              </div>
              <div>
                <span>Latest activity classification</span>
                <strong>{latest ? pressureLabel(latest.pressureIndex) : 'N/A'}</strong>
              </div>
            </div>
          </div>
        </div>

      </div>

      <div className="pressure-section-heading">
        <div>
          <h2>Operational Activity Insights</h2>
          <p>Important patterns identified from the selected historical period</p>
        </div>
      </div>

      <div className="pressure-insights-grid">

        <div className="pressure-insight-card">
          <div className="pressure-insight-icon peak">
            <TrendingUp size={20} />
          </div>
          <span>HIGHEST ACTIVITY</span>
          <h3>{peakPressure ? formatMonth(peakPressure.month) : 'N/A'}</h3>
          <p>
            {peakPressure
              ? `Relative activity index: ${peakPressure.pressureIndex.toFixed(1)}%`
              : 'No data available'}
          </p>
        </div>

        <div className="pressure-insight-card">
          <div className="pressure-insight-icon low">
            <TrendingDown size={20} />
          </div>
          <span>LOWEST ACTIVITY</span>
          <h3>{lowestPressure ? formatMonth(lowestPressure.month) : 'N/A'}</h3>
          <p>
            {lowestPressure
              ? `Relative activity index: ${lowestPressure.pressureIndex.toFixed(1)}%`
              : 'No data available'}
          </p>
        </div>

        <div className="pressure-insight-card">
          <div className="pressure-insight-icon current">
            <Activity size={20} />
          </div>
          <span>LATEST DATA PERIOD</span>
          <h3>{latest ? formatMonth(latest.month) : 'N/A'}</h3>
          <p>
            {latest
              ? `${numberFormat(latest.passengers)} passengers and ${numberFormat(latest.landings)} landings`
              : 'No data available'}
          </p>
        </div>

      </div>

      <div className="pressure-methodology">
        <AlertTriangle size={19} />
        <div>
          <strong>About the Operational Pressure Index</strong>
          <p>
            This is an analytical indicator, not a measured congestion level.
            Passenger and aircraft activity are each normalized against their
            historical maximum. The combined index is the average of those
            two normalized values. Actual congestion also depends on runway
            capacity, flight schedules, delays, terminal occupancy and other
            operational factors not included in these datasets.
          </p>
        </div>
      </div>

      <div className="pressure-footer">
        <span> AEROCAST AI · SFO AIRPORT ANALYTICS</span>
        <span>{mergedData.length} monthly records available</span>
      </div>

    </div>
  )
}

export default OperationalPressure
