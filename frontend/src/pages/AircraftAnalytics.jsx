
import { useEffect, useMemo, useState } from 'react'
import axios from 'axios'

import {
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from 'recharts'

import {
  Plane,
  Weight,
  Activity,
  TrendingUp,
  RefreshCw,
  CalendarDays,
  Gauge,
  ArrowUpRight
} from 'lucide-react'

import './AircraftAnalytics.css'

const API = 'http://127.0.0.1:8000/api/analytics'

const number = value =>
  new Intl.NumberFormat('en-US').format(value || 0)

const compact = value => {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`
  return number(value)
}

const formatMonth = value => {
  if (!value) return ''
  const [year, month] = value.split('-')
  return new Date(Number(year), Number(month) - 1)
    .toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
}

function AircraftAnalytics() {
  const [summary, setSummary] = useState(null)
  const [monthly, setMonthly] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [period, setPeriod] = useState('ALL')

  const loadData = async () => {
    try {
      setLoading(true)
      setError('')

      const [summaryResponse, monthlyResponse] = await Promise.all([
        axios.get(`${API}/aircraft`),
        axios.get(`${API}/aircraft/monthly`)
      ])

      setSummary(summaryResponse.data)

      const records = Array.isArray(monthlyResponse.data)
        ? monthlyResponse.data
        : monthlyResponse.data.monthly_traffic || monthlyResponse.data.monthly_data || []

      setMonthly(records)
    } catch (err) {
      console.error('Aircraft analytics error:', err)
      setError('Unable to load aircraft analytics. Check that the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  const filteredData = useMemo(() => {
    if (period === 'ALL') return monthly

    const years = Number(period)
    return monthly.slice(-years * 12)
  }, [monthly, period])

  const yearlyData = useMemo(() => {
    const grouped = {}

    monthly.forEach(item => {
      const year = item.month.slice(0, 4)

      if (!grouped[year]) {
        grouped[year] = { year, landings: 0, landed_weight: 0 }
      }

      grouped[year].landings += Number(item.landings || 0)
      grouped[year].landed_weight += Number(item.landed_weight || 0)
    })

    return Object.values(grouped)
  }, [monthly])

  const peakMonth = useMemo(() => {
    if (!monthly.length) return null
    return monthly.reduce((max, item) =>
      Number(item.landings) > Number(max.landings) ? item : max
    )
  }, [monthly])

  const latest = monthly[monthly.length - 1]

  const averageWeight = summary?.total_landings
    ? summary.total_landed_weight / summary.total_landings
    : 0

  if (loading) {
    return (
      <div className="aircraft-loading">
        <RefreshCw className="spin" size={28} />
        <p>Loading Aircraft Analytics...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="aircraft-error">
        <p>{error}</p>
        <button onClick={loadData}>Retry</button>
      </div>
    )
  }

  return (
    <div className="aircraft-page">

      <div className="aircraft-heading">
        <div>
          <div className="aircraft-eyebrow">
            <Plane size={15} />
            AIRPORT OPERATIONS / FLEET MOVEMENT
          </div>

          <h1>Aircraft Analytics</h1>

          <p>
            Comprehensive aircraft landing and operational weight analysis
            for San Francisco International Airport.
          </p>
        </div>

        <button className="aircraft-refresh" onClick={loadData}>
          <RefreshCw size={16} />
          Refresh Data
        </button>
      </div>

      <div className="aircraft-data-note">
        <Activity size={15} />
        Historical dataset · Latest available record: {summary?.latest_month || 'N/A'}
      </div>

      <div className="aircraft-kpi-grid">

        <div className="aircraft-kpi">
          <div className="aircraft-kpi-top">
            <span>Total Aircraft Landings</span>
            <div className="aircraft-icon cyan">
              <Plane size={21} />
            </div>
          </div>
          <h2>{number(summary?.total_landings)}</h2>
          <p><TrendingUp size={14} /> Total recorded landings</p>
        </div>

        <div className="aircraft-kpi">
          <div className="aircraft-kpi-top">
            <span>Total Landed Weight</span>
            <div className="aircraft-icon purple">
              <Weight size={21} />
            </div>
          </div>
          <h2>{compact(summary?.total_landed_weight)}</h2>
          <p><Activity size={14} /> Dataset weight units</p>
        </div>

        <div className="aircraft-kpi">
          <div className="aircraft-kpi-top">
            <span>Average Monthly Landings</span>
            <div className="aircraft-icon blue">
              <Gauge size={21} />
            </div>
          </div>
          <h2>{number(Math.round(summary?.average_monthly_landings || 0))}</h2>
          <p><CalendarDays size={14} /> Average per month</p>
        </div>

        <div className="aircraft-kpi">
          <div className="aircraft-kpi-top">
            <span>Latest Monthly Landings</span>
            <div className="aircraft-icon green">
              <ArrowUpRight size={21} />
            </div>
          </div>
          <h2>{number(summary?.latest_landings)}</h2>
          <p><CalendarDays size={14} /> {summary?.latest_month}</p>
        </div>

      </div>

      <div className="aircraft-section-title">
        <div>
          <h2>Aircraft Movement Analysis</h2>
          <p>Historical aircraft landing activity over time</p>
        </div>

        <div className="aircraft-periods">
          {['ALL', '10', '5', '3', '1'].map(value => (
            <button
              key={value}
              className={period === value ? 'active' : ''}
              onClick={() => setPeriod(value)}
            >
              {value === 'ALL' ? 'All Time' : `${value}Y`}
            </button>
          ))}
        </div>
      </div>

      <div className="aircraft-chart-grid">

        <div className="aircraft-chart-card wide">
          <div className="aircraft-chart-heading">
            <div>
              <h3>Monthly Aircraft Landings</h3>
              <p>Number of aircraft landing at SFO</p>
            </div>
            <span className="chart-unit">LANDINGS</span>
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={filteredData}>
              <defs>
                <linearGradient id="landingGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00D9FF" stopOpacity={0.35} />
                  <stop offset="95%" stopColor="#00D9FF" stopOpacity={0.01} />
                </linearGradient>
              </defs>

              <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />

              <XAxis
                dataKey="month"
                tickFormatter={formatMonth}
                stroke="var(--chart-muted)"
                minTickGap={35}
                tick={{ fontSize: 11 }}
              />

              <YAxis
                stroke="var(--chart-muted)"
                tickFormatter={compact}
                tick={{ fontSize: 11 }}
              />

              <Tooltip
                labelFormatter={formatMonth}
                formatter={value => [number(value), 'Landings']}
                contentStyle={{
                  background: 'var(--chart-tooltip)',
                  border: '1px solid var(--chart-border)',
                  borderRadius: 10
                }}
              />

              <Area
                type="monotone"
                dataKey="landings"
                stroke="#00D9FF"
                strokeWidth={2.5}
                fill="url(#landingGradient)"
                activeDot={{ r: 5 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="aircraft-chart-card">
          <div className="aircraft-chart-heading">
            <div>
              <h3>Landed Weight Trend</h3>
              <p>Monthly aircraft weight</p>
            </div>
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={filteredData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
              <XAxis
                dataKey="month"
                tickFormatter={formatMonth}
                stroke="var(--chart-muted)"
                minTickGap={35}
                tick={{ fontSize: 10 }}
              />
              <YAxis
                stroke="var(--chart-muted)"
                tickFormatter={compact}
                tick={{ fontSize: 10 }}
              />
              <Tooltip
                labelFormatter={formatMonth}
                formatter={value => [number(value), 'Landed Weight']}
              />
              <Line
                type="monotone"
                dataKey="landed_weight"
                stroke="#647BFF"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="aircraft-chart-card">
          <div className="aircraft-chart-heading">
            <div>
              <h3>Year-wise Aircraft Activity</h3>
              <p>Annual total aircraft landings</p>
            </div>
            <span className="chart-unit">YEARLY TOTAL</span>
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={yearlyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
              <XAxis dataKey="year" stroke="var(--chart-muted)" tick={{ fontSize: 11 }} />
              <YAxis stroke="var(--chart-muted)" tickFormatter={compact} />
              <Tooltip formatter={value => [number(value), 'Landings']} />
              <Bar dataKey="landings" fill="#A78BFA" radius={[5, 5, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="aircraft-chart-card">
          <div className="aircraft-chart-heading">
            <div>
              <h3>Year-wise Landed Weight</h3>
              <p>Total weight by year</p>
            </div>
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={yearlyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
              <XAxis dataKey="year" stroke="var(--chart-muted)" tick={{ fontSize: 10 }} />
              <YAxis stroke="var(--chart-muted)" tickFormatter={compact} />
              <Tooltip formatter={value => [number(value), 'Weight']} />
              <Bar dataKey="landed_weight" fill="#34D399" radius={[5, 5, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

      </div>

      <div className="aircraft-insight-grid">

        <div className="aircraft-insight">
          <span>PEAK LANDING MONTH</span>
          <h2>{peakMonth ? formatMonth(peakMonth.month) : 'N/A'}</h2>
          <p>{number(peakMonth?.landings)} aircraft landings</p>
        </div>

        <div className="aircraft-insight">
          <span>LATEST DATA PERIOD</span>
          <h2>{latest ? formatMonth(latest.month) : 'N/A'}</h2>
          <p>{number(latest?.landings)} aircraft landings</p>
        </div>

        <div className="aircraft-insight">
          <span>AVERAGE WEIGHT PER LANDING</span>
          <h2>{compact(averageWeight)}</h2>
          <p>Calculated from total landed weight</p>
        </div>

      </div>

    </div>
  )
}

export default AircraftAnalytics
