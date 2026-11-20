import './PassengerAnalytics.css'
import { useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import {
  ResponsiveContainer,
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  BarChart, Bar, PieChart, Pie, Cell, Legend
} from 'recharts'
import {
  Users, TrendingUp, Plane, Globe, Activity,
  RefreshCw, CalendarDays, ArrowUpRight
} from 'lucide-react'

const API = 'http://127.0.0.1:8000/api/analytics/passengers/detailed'

const COLORS = ['#00D9FF', '#647BFF', '#A78BFA', '#38BDF8', '#F59E0B', '#2DD4BF']

const number = (value) => new Intl.NumberFormat('en-US').format(Math.round(value || 0))

const compact = (value) => {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`
  if (value >= 1e3) return `${(value / 1e3).toFixed(0)}K`
  return number(value)
}

function Card({ children, className = '' }) {
  return <div className={`pa-card ${className}`}>{children}</div>
}

function ChartTitle({ title, subtitle }) {
  return (
    <div className="pa-chart-heading">
      <div>
        <h3>{title}</h3>
        <p>{subtitle}</p>
      </div>
    </div>
  )
}

function PassengerAnalytics() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [period, setPeriod] = useState('all')

  const fetchData = async () => {
    try {
      setLoading(true)
      setError('')
      const response = await axios.get(API)
      setData(response.data)
    } catch (err) {
      console.error('Passenger analytics error:', err)
      setError('Unable to load passenger analytics. Check whether the FastAPI backend is running.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

  const monthlyData = useMemo(() => {
    if (!data?.monthly_traffic) return []
    let records = data.monthly_traffic

    if (period !== 'all') {
      const years = Number(period)
      records = records.slice(-years * 12)
    }

    return records
  }, [data, period])

  if (loading) {
    return <div className="pa-message"><RefreshCw className="pa-spin" /> Loading passenger analytics...</div>
  }

  if (error) {
    return (
      <div className="pa-message pa-error">
        <p>{error}</p>
        <button onClick={fetchData}>Retry</button>
      </div>
    )
  }

  if (!data) return null

  const domestic = data.domestic_international?.find(x => x.category === 'Domestic')?.passengers || 0
  const international = data.domestic_international?.find(x => x.category === 'International')?.passengers || 0
  const activityTotal = (data.activity_distribution || []).reduce((sum, item) => sum + item.passengers, 0)

  const peak = data.peak_month || {}
  const lowest = data.lowest_month || {}

  const topAirlines = (data.airline_rankings || []).slice(0, 8)
  const terminals = data.terminal_distribution || []
  const boardingAreas = data.boarding_area_distribution || []
  const activities = data.activity_distribution || []
  const seasons = data.seasonal_traffic || []

  return (
    <div className="passenger-page">
     
      <div className="pa-header">
        <div>
          <h1>Passenger Analytics</h1>
          <p>Detailed passenger traffic intelligence · San Francisco International Airport</p>
        </div>
        <div className="pa-actions">
          <select className="pa-select" value={period} onChange={e => setPeriod(e.target.value)}>
            <option value="all">All available data</option>
            <option value="5">Last 5 years</option>
            <option value="3">Last 3 years</option>
            <option value="1">Last 12 months</option>
          </select>
          <button className="pa-refresh" onClick={fetchData}><RefreshCw size={15} /> Refresh</button>
        </div>
      </div>

      <div className="pa-kpis">
        <Card>
          <div className="pa-kpi-top"><Users size={20} /><div className="pa-kpi-icon"><Users size={19} /></div></div>
          <div className="pa-kpi-label">Total Passengers</div>
          <div className="pa-kpi-value">{compact(data.total_passengers)}</div>
          <div className="pa-kpi-note">{number(data.total_passengers)} recorded passengers</div>
        </Card>
        <Card>
          <div className="pa-kpi-top"><Globe size={20} /><div className="pa-kpi-icon"><Globe size={19} /></div></div>
          <div className="pa-kpi-label">Domestic Passengers</div>
          <div className="pa-kpi-value">{compact(domestic)}</div>
          <div className="pa-kpi-note">{(domestic / data.total_passengers * 100).toFixed(1)}% of total traffic</div>
        </Card>
        <Card>
          <div className="pa-kpi-top"><Plane size={20} /><div className="pa-kpi-icon"><Plane size={19} /></div></div>
          <div className="pa-kpi-label">International Passengers</div>
          <div className="pa-kpi-value">{compact(international)}</div>
          <div className="pa-kpi-note">{(international / data.total_passengers * 100).toFixed(1)}% of total traffic</div>
        </Card>
        <Card>
          <div className="pa-kpi-top"><TrendingUp size={20} /><div className="pa-kpi-icon"><TrendingUp size={19} /></div></div>
          <div className="pa-kpi-label">Peak Monthly Traffic</div>
          <div className="pa-kpi-value">{compact(peak.passengers)}</div>
          <div className="pa-kpi-note">{peak.month || 'Not available'}</div>
        </Card>
      </div>

      <div className="pa-insight">
        <h3><Activity size={15} style={{display:'inline', marginRight:7}} /> Passenger Traffic Overview</h3>
        <p>
          The dataset contains {number(data.total_passengers)} passenger records.
          The highest recorded month was {peak.month} with {number(peak.passengers)} passengers.
          The lowest recorded month was {lowest.month} with {number(lowest.passengers)} passengers.
        </p>
      </div>

      <div className="pa-grid">
        <Card>
          <ChartTitle title="Monthly Passenger Traffic" subtitle="Historical passenger volume by month" />
          <div className="pa-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={monthlyData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#94a3b833" />
                <XAxis dataKey="month" minTickGap={35} />
                <YAxis tickFormatter={compact} width={55} />
                <Tooltip formatter={v => number(v)} />
                <Line type="monotone" dataKey="passengers" name="Passengers" stroke="#84cc16" strokeWidth={2.5} dot={false} activeDot={{r:5}} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card>
          <ChartTitle title="Domestic vs International" subtitle="Passenger distribution by travel category" />
          <div className="pa-chart">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={data.domestic_international} dataKey="passengers" nameKey="category" cx="50%" cy="48%" outerRadius={90} innerRadius={53} paddingAngle={3}>
                  {data.domestic_international.map((item, i) => <Cell key={item.category} fill={COLORS[i]} />)}
                </Pie>
                <Tooltip formatter={v => number(v)} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card>
          <ChartTitle title="Year-wise Passenger Growth" subtitle="Annual passenger volume" />
          <div className="pa-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.yearly_traffic}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#94a3b833" />
                <XAxis dataKey="year" minTickGap={25} />
                <YAxis tickFormatter={compact} width={55} />
                <Tooltip formatter={v => number(v)} />
                <Bar dataKey="passengers" name="Passengers" fill="#0ea5e9" radius={[5,5,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card>
          <ChartTitle title="Top Airlines" subtitle="Airlines ranked by recorded passenger volume" />
          <div className="pa-rank-list">
            {topAirlines.map((item, i) => (
              <div key={item.airline}>
                <div className="pa-rank-row">
                  <span className="pa-rank-number">{String(i + 1).padStart(2, '0')}</span>
                  <span className="pa-rank-name">{item.airline}</span>
                  <span className="pa-rank-value">{compact(item.passengers)}</span>
                </div>
                <div className="pa-progress">
                  <span style={{width:`${item.passengers / (topAirlines[0]?.passengers || 1) * 100}%`}} />
                </div>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <ChartTitle title="Terminal Distribution" subtitle="Passenger volume by terminal" />
          <div className="pa-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={terminals} layout="vertical" margin={{left:15,right:15}}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#94a3b833" />
                <XAxis type="number" tickFormatter={compact} />
                <YAxis type="category" dataKey="terminal" width={95} />
                <Tooltip formatter={v => number(v)} />
                <Bar dataKey="passengers" name="Passengers" fill="#8b5cf6" radius={[0,5,5,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card>
          <ChartTitle title="Boarding Area Distribution" subtitle="Passenger volume by boarding area" />
          <div className="pa-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={boardingAreas}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#94a3b833" />
                <XAxis dataKey="boarding_area" />
                <YAxis tickFormatter={compact} width={55} />
                <Tooltip formatter={v => number(v)} />
                <Bar dataKey="passengers" name="Passengers" fill="#f59e0b" radius={[5,5,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card>
          <ChartTitle title="Enplaned vs Deplaned" subtitle="Passenger activity distribution" />
          <div className="pa-chart">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={activities} dataKey="passengers" nameKey="activity" cx="50%" cy="48%" outerRadius={90} innerRadius={48}>
                  {activities.map((item, i) => <Cell key={item.activity} fill={COLORS[i]} />)}
                </Pie>
                <Tooltip formatter={v => number(v)} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card>
          <ChartTitle title="Seasonal Passenger Trends" subtitle="Average passenger volume by calendar month" />
          <div className="pa-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={seasons}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#94a3b833" />
                <XAxis dataKey="month_name" tick={{fontSize:10}} interval={0} />
                <YAxis tickFormatter={compact} width={55} />
                <Tooltip formatter={v => number(v)} />
                <Bar dataKey="average_passengers" name="Average passengers" fill="#14b8a6" radius={[5,5,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card>
        <ChartTitle title="Passenger Activity Summary" subtitle="Recorded passenger activity totals" />
        <div className="pa-peak-grid">
          <div className="pa-peak-box">
            <span>Peak month</span>
            <strong>{peak.month}</strong>
            <small>{number(peak.passengers)} passengers</small>
          </div>
          <div className="pa-peak-box">
            <span>Lowest month</span>
            <strong>{lowest.month}</strong>
            <small>{number(lowest.passengers)} passengers</small>
          </div>
        </div>
        <p style={{fontSize:11, color:'var(--text-secondary,#718096)', marginTop:15}}>
          Activity total: {number(activityTotal)} · Values are calculated from the connected SFO dataset.
        </p>
      </Card>
    </div>
  )
}

export default PassengerAnalytics
