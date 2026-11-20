
import { useEffect, useMemo, useState } from 'react'
import axios from 'axios'

import {
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip
} from 'recharts'

import {
  Package,
  Weight,
  TrendingUp,
  Activity,
  RefreshCw,
  CalendarDays,
  ArrowUpRight,
  ArrowDownRight,
  Plane
} from 'lucide-react'

import './CargoAnalytics.css'

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
    .toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
}

function CargoAnalytics() {
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
        axios.get(`${API}/cargo`),
        axios.get(`${API}/cargo/monthly`)
      ])

      setSummary(summaryResponse.data)

      const records = Array.isArray(monthlyResponse.data)
        ? monthlyResponse.data
        : monthlyResponse.data.monthly_traffic ||
          monthlyResponse.data.monthly_data ||
          []

      setMonthly(records)
    } catch (err) {
      console.error('Cargo analytics error:', err)
      setError('Unable to load cargo analytics. Check that the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  const filteredData = useMemo(() => {
    if (period === 'ALL') return monthly
    return monthly.slice(-Number(period) * 12)
  }, [monthly, period])

  const yearlyData = useMemo(() => {
    const grouped = {}

    monthly.forEach(item => {
      const year = item.month.slice(0, 4)

      if (!grouped[year]) {
        grouped[year] = { year, cargo: 0 }
      }

      grouped[year].cargo += Number(item.cargo_weight_lbs || 0)
    })

    return Object.values(grouped)
  }, [monthly])

  const peakMonth = useMemo(() => {
    if (!monthly.length) return null

    return monthly.reduce((max, item) =>
      Number(item.cargo_weight_lbs) > Number(max.cargo_weight_lbs)
        ? item
        : max
    )
  }, [monthly])

  const lowestMonth = useMemo(() => {
    if (!monthly.length) return null

    return monthly.reduce((min, item) =>
      Number(item.cargo_weight_lbs) < Number(min.cargo_weight_lbs)
        ? item
        : min
    )
  }, [monthly])

  const latest = monthly[monthly.length - 1]

  const totalCargo =
    Number(summary?.total_cargo_weight_lbs) ||
    monthly.reduce((sum, item) => sum + Number(item.cargo_weight_lbs || 0), 0)

  const averageCargo =
    Number(summary?.average_monthly_cargo_lbs) ||
    (monthly.length ? totalCargo / monthly.length : 0)

  if (loading) {
    return (
      <div className="cargo-loading">
        <RefreshCw className="cargo-spin" size={28} />
        <p>Loading Cargo Analytics...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="cargo-error">
        <p>{error}</p>
        <button onClick={loadData}>Retry</button>
      </div>
    )
  }

  return (
    <div className="cargo-page">

      <div className="cargo-heading">
        <div>
          <div className="cargo-eyebrow">
            <Package size={15} />
            AIRPORT OPERATIONS / FREIGHT ANALYTICS
          </div>

          <h1>Cargo Analytics</h1>

          <p>
            Comprehensive air freight volume analysis and cargo traffic
            patterns for San Francisco International Airport.
          </p>
        </div>

        <button className="cargo-refresh" onClick={loadData}>
          <RefreshCw size={16} />
          Refresh Data
        </button>
      </div>

      <div className="cargo-data-note">
        <Activity size={16} />
        Historical cargo dataset · Latest available record:{' '}
        {summary?.latest_month || latest?.month || 'N/A'}
      </div>

      <div className="cargo-kpi-grid">

        <div className="cargo-kpi">
          <div className="cargo-kpi-top">
            <span>Total Cargo Weight</span>
            <div className="cargo-icon lime">
              <Weight size={21} />
            </div>
          </div>
          <h2>{compact(totalCargo)}</h2>
          <p><Package size={14} /> Total recorded cargo in lbs</p>
        </div>

        <div className="cargo-kpi">
          <div className="cargo-kpi-top">
            <span>Average Monthly Cargo</span>
            <div className="cargo-icon green">
              <TrendingUp size={21} />
            </div>
          </div>
          <h2>{compact(averageCargo)}</h2>
          <p><CalendarDays size={14} /> Average cargo per month</p>
        </div>

        <div className="cargo-kpi">
          <div className="cargo-kpi-top">
            <span>Latest Monthly Cargo</span>
            <div className="cargo-icon blue">
              <ArrowUpRight size={21} />
            </div>
          </div>
          <h2>{compact(Number(summary?.latest_cargo_weight_lbs || latest?.cargo_weight_lbs || 0))}</h2>
          <p><CalendarDays size={14} /> {summary?.latest_month || latest?.month || 'N/A'}</p>
        </div>

        <div className="cargo-kpi">
          <div className="cargo-kpi-top">
            <span>Peak Cargo Volume</span>
            <div className="cargo-icon orange">
              <Plane size={21} />
            </div>
          </div>
          <h2>{compact(Number(summary?.highest_cargo_weight_lbs || peakMonth?.cargo_weight_lbs || 0))}</h2>
          <p><TrendingUp size={14} /> {summary?.highest_cargo_month || peakMonth?.month || 'N/A'}</p>
        </div>

      </div>

      <div className="cargo-section-title">
        <div>
          <h2>Cargo Traffic Analysis</h2>
          <p>Historical air freight movement and volume trends</p>
        </div>

        <div className="cargo-periods">
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

      <div className="cargo-chart-grid">

        <div className="cargo-chart-card wide">
          <div className="cargo-chart-heading">
            <div>
              <h3>Monthly Cargo Traffic</h3>
              <p>Total monthly cargo weight handled at SFO</p>
            </div>
            <span className="cargo-chart-unit">CARGO / LBS</span>
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={filteredData}>
              <defs>
                <linearGradient id="cargoGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#84CC16" stopOpacity={0.35} />
                  <stop offset="95%" stopColor="#84CC16" stopOpacity={0.02} />
                </linearGradient>
              </defs>

              <CartesianGrid strokeDasharray="3 3" stroke="var(--cargo-chart-grid)" />

              <XAxis
                dataKey="month"
                tickFormatter={formatMonth}
                stroke="var(--cargo-chart-muted)"
                minTickGap={35}
                tick={{ fontSize: 11 }}
              />

              <YAxis
                stroke="var(--cargo-chart-muted)"
                tickFormatter={compact}
                tick={{ fontSize: 11 }}
              />

              <Tooltip
                labelFormatter={formatMonth}
                formatter={value => [`${number(value)} lbs`, 'Cargo Weight']}
                contentStyle={{
                  background: 'var(--cargo-tooltip)',
                  border: '1px solid var(--cargo-border)',
                  borderRadius: 10
                }}
              />

              <Area
                type="monotone"
                dataKey="cargo_weight_lbs"
                stroke="#65A30D"
                strokeWidth={2.5}
                fill="url(#cargoGradient)"
                activeDot={{ r: 5 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="cargo-chart-card">
          <div className="cargo-chart-heading">
            <div>
              <h3>Cargo Volume Trend</h3>
              <p>Monthly freight weight variation</p>
            </div>
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={filteredData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--cargo-chart-grid)" />
              <XAxis
                dataKey="month"
                tickFormatter={formatMonth}
                stroke="var(--cargo-chart-muted)"
                minTickGap={35}
                tick={{ fontSize: 10 }}
              />
              <YAxis
                stroke="var(--cargo-chart-muted)"
                tickFormatter={compact}
                tick={{ fontSize: 10 }}
              />
              <Tooltip
                labelFormatter={formatMonth}
                formatter={value => [`${number(value)} lbs`, 'Cargo Weight']}
              />
              <Line
                type="monotone"
                dataKey="cargo_weight_lbs"
                stroke="#84CC16"
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 5 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="cargo-chart-card">
          <div className="cargo-chart-heading">
            <div>
              <h3>Year-wise Cargo Volume</h3>
              <p>Annual total cargo weight</p>
            </div>
            <span className="cargo-chart-unit">YEARLY TOTAL</span>
          </div>

          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={yearlyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--cargo-chart-grid)" />
              <XAxis
                dataKey="year"
                stroke="var(--cargo-chart-muted)"
                tick={{ fontSize: 10 }}
              />
              <YAxis
                stroke="var(--cargo-chart-muted)"
                tickFormatter={compact}
              />
              <Tooltip formatter={value => [`${number(value)} lbs`, 'Cargo Weight']} />
              <Bar
                dataKey="cargo"
                fill="#84CC16"
                radius={[5, 5, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

      </div>

      <div className="cargo-insight-grid">

        <div className="cargo-insight">
          <span>HIGHEST CARGO MONTH</span>
          <h2>{peakMonth ? formatMonth(peakMonth.month) : 'N/A'}</h2>
          <p>{number(peakMonth?.cargo_weight_lbs)} lbs recorded</p>
        </div>

        <div className="cargo-insight">
          <span>LOWEST CARGO MONTH</span>
          <h2>{lowestMonth ? formatMonth(lowestMonth.month) : 'N/A'}</h2>
          <p>{number(lowestMonth?.cargo_weight_lbs)} lbs recorded</p>
        </div>

        <div className="cargo-insight">
          <span>LATEST DATA PERIOD</span>
          <h2>{latest ? formatMonth(latest.month) : 'N/A'}</h2>
          <p>{number(latest?.cargo_weight_lbs)} lbs handled</p>
        </div>

      </div>

    </div>
  )
}

export default CargoAnalytics
