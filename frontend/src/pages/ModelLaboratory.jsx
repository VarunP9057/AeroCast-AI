import { useEffect, useMemo, useState } from 'react'
import {
  BrainCircuit, Database, FlaskConical, Activity,
  RefreshCw, AlertCircle, Trophy, BarChart3
} from 'lucide-react'
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip
} from 'recharts'
import api from '../services/api'
import './ModelLaboratory.css'

const datasets = [
  'All Datasets',
  'Passenger',
  'Aircraft',
  'Cargo',
  'Terminal',
  'Operational Pressure'
]

const formatNumber = value => {
  if (value === null || value === undefined) return '—'
  return Number(value).toLocaleString('en-US', {
    maximumFractionDigits: 2
  })
}

const formatR2 = value => {
  if (value === null || value === undefined) return '—'
  return `${(Number(value) * 100).toFixed(1)}%`
}

export default function ModelLaboratory() {
  const [overview, setOverview] = useState(null)
  const [results, setResults] = useState([])
  const [selectedDataset, setSelectedDataset] = useState('All Datasets')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const loadData = async (dataset = selectedDataset) => {
    try {
      setLoading(true)
      setError('')

      const [overviewResponse, resultsResponse] = await Promise.all([
        api.get('/api/model-lab/overview'),
        api.get('/api/model-lab/results', {
          params: dataset === 'All Datasets' ? {} : { dataset }
        })
      ])

      setOverview(overviewResponse.data)
      setResults(resultsResponse.data.results || [])
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        'Unable to load Model Laboratory data. Check that the backend is running.'
      )
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData('All Datasets')
  }, [])

  const handleDatasetChange = event => {
    const dataset = event.target.value
    setSelectedDataset(dataset)
    loadData(dataset)
  }

  const annResults = useMemo(
    () => results.filter(item => item.model.toLowerCase().includes('ann')),
    [results]
  )

  const bestModel = useMemo(() => {
    const valid = results.filter(item => item.r2 !== null)
    if (!valid.length) return null
    return [...valid].sort((a, b) => b.r2 - a.r2)[0]
  }, [results])

  const chartData = useMemo(() => {
    return results
      .filter(item => item.r2 !== null)
      .slice(0, 12)
      .map(item => ({
        name: item.model.length > 20
          ? `${item.model.substring(0, 20)}…`
          : item.model,
        r2: Number((item.r2 * 100).toFixed(2))
      }))
  }, [results])

  return (
    <div className="model-lab-page">
      <div className="model-lab-header">
        <div>
          <div className="model-lab-eyebrow">
            <BrainCircuit size={15} />
            AEROCAST AI / INTELLIGENCE CENTER
          </div>
          <h1>Model <span>Laboratory</span></h1>
          <p>
            Explore trained machine learning models, compare experiments,
            and evaluate airport forecasting performance.
          </p>
        </div>

        <button
          className="model-refresh-button"
          onClick={() => loadData()}
          disabled={loading}
        >
          <RefreshCw size={15} className={loading ? 'refresh-spin' : ''} />
          Refresh Data
        </button>
      </div>

      {error && (
        <div className="model-error">
          <AlertCircle size={18} />
          {error}
        </div>
      )}

      <div className="model-filter-row">
        <div>
          <label htmlFor="model-dataset">SELECT DATASET</label>
          <select
            id="model-dataset"
            value={selectedDataset}
            onChange={handleDatasetChange}
          >
            {datasets.map(dataset => (
              <option key={dataset} value={dataset}>{dataset}</option>
            ))}
          </select>
        </div>
        <div className="model-live-status">
          <span />
          LIVE REPORT DATA
        </div>
      </div>

      <div className="model-kpi-grid">
        <div className="model-kpi-card">
          <div className="model-kpi-icon cyan"><Database size={20} /></div>
          <div>
            <span>Evaluation Results</span>
            <strong>{loading ? '...' : formatNumber(results.length)}</strong>
            <small>Recorded model evaluations</small>
          </div>
        </div>

        <div className="model-kpi-card">
          <div className="model-kpi-icon purple"><BrainCircuit size={20} /></div>
          <div>
            <span>ANN Results</span>
            <strong>{loading ? '...' : formatNumber(annResults.length)}</strong>
            <small>ANN evaluation entries</small>
          </div>
        </div>

        <div className="model-kpi-card">
          <div className="model-kpi-icon blue"><FlaskConical size={20} /></div>
          <div>
            <span>Trained ANN Artifacts</span>
            <strong>{loading ? '...' : formatNumber(overview?.trained_ann_artifacts)}</strong>
            <small>Keras model files found</small>
          </div>
        </div>

        <div className="model-kpi-card">
          <div className="model-kpi-icon green"><Activity size={20} /></div>
          <div>
            <span>Best Recorded R²</span>
            <strong>{loading ? '...' : formatR2(bestModel?.r2)}</strong>
            <small>{bestModel?.model || 'No result available'}</small>
          </div>
        </div>
      </div>

      <div className="model-section-heading">
        <div>
          <span>MODEL EVALUATION</span>
          <h2>Performance Analysis</h2>
        </div>
        <span className="model-result-count">
          {results.length} evaluation records
        </span>
      </div>

      <div className="model-content-grid">
        <section className="model-panel model-chart-panel">
          <div className="model-panel-heading">
            <div>
              <BarChart3 size={18} />
              <h3>Model R² Comparison</h3>
            </div>
            <span>R² SCORE (%)</span>
          </div>

          {chartData.length ? (
            <div className="model-chart">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 15, right: 12, left: 0, bottom: 55 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--ml-grid)" />
                  <XAxis
                    dataKey="name"
                    angle={-30}
                    textAnchor="end"
                    interval={0}
                    tick={{ fontSize: 10, fill: 'var(--ml-muted)' }}
                    height={75}
                  />
                  <YAxis
                    domain={['auto', 100]}
                    tick={{ fontSize: 10, fill: 'var(--ml-muted)' }}
                  />
                  <Tooltip
                    formatter={value => [`${value}%`, 'R² Score']}
                    contentStyle={{
                      background: 'var(--ml-card)',
                      border: '1px solid var(--ml-border)',
                      borderRadius: 8,
                      color: 'var(--ml-text)'
                    }}
                  />
                  <Bar dataKey="r2" fill="#00D9FF" radius={[5, 5, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="model-empty">
              {loading ? 'Loading evaluation data...' : 'No R² results available for this dataset.'}
            </div>
          )}
        </section>

        <section className="model-panel model-best-panel">
          <div className="model-panel-heading">
            <div>
              <Trophy size={18} />
              <h3>Top Recorded Result</h3>
            </div>
          </div>

          {bestModel ? (
            <>
              <div className="model-best-score">{formatR2(bestModel.r2)}</div>
              <div className="model-best-label">R² SCORE</div>
              <h3 className="model-best-name">{bestModel.model}</h3>
              <p>{bestModel.dataset} · {bestModel.experiment}</p>
              <div className="model-best-metrics">
                <div><span>MAE</span><strong>{formatNumber(bestModel.mae)}</strong></div>
                <div><span>RMSE</span><strong>{formatNumber(bestModel.rmse)}</strong></div>
              </div>
            </>
          ) : (
            <div className="model-empty">No evaluated model available.</div>
          )}
        </section>
      </div>

      <div className="model-section-heading">
        <div>
          <span>EXPERIMENT DATABASE</span>
          <h2>Model Evaluation Records</h2>
        </div>
      </div>

      <section className="model-panel model-table-panel">
        {loading ? (
          <div className="model-empty">Loading model evaluation records...</div>
        ) : results.length ? (
          <div className="model-table-scroll">
            <table className="model-table">
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Model</th>
                  <th>Experiment</th>
                  <th>MAE</th>
                  <th>RMSE</th>
                  <th>R²</th>
                  <th>MAPE</th>
                  <th>WAPE</th>
                </tr>
              </thead>
              <tbody>
                {results.map((item, index) => (
                  <tr key={`${item.dataset}-${item.experiment}-${item.model}-${index}`}>
                    <td><span className="model-dataset-tag">{item.dataset}</span></td>
                    <td className="model-name-cell">{item.model}</td>
                    <td>{item.experiment}</td>
                    <td>{formatNumber(item.mae)}</td>
                    <td>{formatNumber(item.rmse)}</td>
                    <td className="model-r2-cell">{formatR2(item.r2)}</td>
                    <td>{formatNumber(item.mape)}</td>
                    <td>{formatNumber(item.wape)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="model-empty">No evaluation records found.</div>
        )}
      </section>

      <div className="model-lab-footer">
        <span> AEROCAST AI · MODEL INTELLIGENCE</span>
        <span>Metrics loaded from project evaluation reports</span>
      </div>
    </div>
  )
}
