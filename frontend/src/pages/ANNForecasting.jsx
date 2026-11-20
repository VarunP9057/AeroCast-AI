import { useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import { Canvas } from '@react-three/fiber'
import { Float, Stars } from '@react-three/drei'
import { motion } from 'framer-motion'
import {
    ResponsiveContainer,
    AreaChart,
    Area,
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip
} from 'recharts'
import {
    BrainCircuit,
    Plane,
    PlaneTakeoff,
    Activity,
    Users,
    CalendarDays,
    RefreshCw,
    Sparkles,
    Cpu,
    ArrowUpRight,
    CircleCheck,
    AlertCircle,
    Gauge,
    Send,
    Database,
    Layers3
} from 'lucide-react'
import './ANNForecasting.css'

const API = 'http://127.0.0.1:8000/api'

const number = value =>
    new Intl.NumberFormat('en-US', {
        maximumFractionDigits: 0
    }).format(Number(value) || 0)

const compact = value => {
    const n = Number(value) || 0
    if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`
    if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`
    return number(n)
}

const monthName = value => {
    if (!value) return 'N/A'
    const [year, month] = value.split('-')
    return new Date(Number(year), Number(month) - 1)
        .toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
}

const nextMonth = value => {
    const [year, month] = value.split('-').map(Number)
    const date = new Date(year, month, 1)
    return {
        year: date.getFullYear(),
        month: date.getMonth() + 1
    }
}

function AirportScene() {
    return (
        <Canvas camera={{ position: [0, 7, 15], fov: 48 }}>
            <fog attach="fog" args={['#061713', 12, 34]} />
            <ambientLight intensity={1.2} />
            <directionalLight position={[5, 10, 5]} intensity={2} color="#8BFFD5" />
            <pointLight position={[-7, 3, -4]} intensity={18} color="#00E6A0" />
            <Stars radius={45} depth={35} count={900} factor={3} fade speed={0.4} />

            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.8, 0]}>
                <planeGeometry args={[45, 45]} />
                <meshStandardMaterial color="#071B16" roughness={0.8} />
            </mesh>

            <mesh position={[0, -0.72, -5]}>
                <boxGeometry args={[22, 0.12, 13]} />
                <meshStandardMaterial color="#102D26" roughness={0.7} />
            </mesh>

            {Array.from({ length: 13 }).map((_, i) => (
                <mesh key={i} position={[-9 + i * 1.5, -0.64, -5]}>
                    <boxGeometry args={[0.65, 0.025, 0.12]} />
                    <meshBasicMaterial color="#8BFFD5" />
                </mesh>
            ))}

            {[-1, 1].map((side, i) => (
                <group key={i} position={[side * 8, 0, -8]}>
                    <mesh position={[0, 1.1, 0]}>
                        <boxGeometry args={[5, 2.2, 4]} />
                        <meshStandardMaterial color="#12382E" metalness={0.5} roughness={0.3} />
                    </mesh>
                    <mesh position={[0, 2.25, 0]}>
                        <boxGeometry args={[5.2, 0.08, 4.2]} />
                        <meshBasicMaterial color="#00E6A0" />
                    </mesh>
                    {Array.from({ length: 4 }).map((_, j) => (
                        <mesh key={j} position={[-1.8 + j * 1.2, 1.1, 2.04]}>
                            <boxGeometry args={[0.65, 0.65, 0.04]} />
                            <meshBasicMaterial color="#8BFFD5" />
                        </mesh>
                    ))}
                </group>
            ))}

            <Float speed={1.3} rotationIntensity={0.08} floatIntensity={0.35}>
                <group position={[0, 2.2, -1]} rotation={[0, -0.25, 0]}>
                    <mesh>
                        <boxGeometry args={[0.8, 0.42, 5.4]} />
                        <meshStandardMaterial color="#EAF4FF" metalness={0.65} roughness={0.25} />
                    </mesh>
                    <mesh position={[0, 0.03, -0.1]}>
                        <boxGeometry args={[8, 0.09, 1.15]} />
                        <meshStandardMaterial color="#00A982" metalness={0.5} roughness={0.3} />
                    </mesh>
                    <mesh position={[0, 0.12, 2.25]}>
                        <coneGeometry args={[0.42, 1.1, 16]} />
                        <meshStandardMaterial color="#EAF4FF" />
                    </mesh>
                    <mesh position={[0, 0.3, -2.35]}>
                        <boxGeometry args={[2.1, 0.08, 0.7]} />
                        <meshStandardMaterial color="#00A982" />
                    </mesh>
                    <mesh position={[0, -0.12, 0]}>
                        <boxGeometry args={[0.18, 0.12, 1.8]} />
                        <meshStandardMaterial color="#647BFF" />
                    </mesh>
                </group>
            </Float>
        </Canvas>
    )
}

function ANNForecasting() {
    const [monthly, setMonthly] = useState([])
    const [history, setHistory] = useState(Array(12).fill(''))
    const [targetYear, setTargetYear] = useState(new Date().getFullYear())
    const [targetMonth, setTargetMonth] = useState(new Date().getMonth() + 1)
    const [prediction, setPrediction] = useState(null)
    const [loading, setLoading] = useState(true)
    const [predicting, setPredicting] = useState(false)
    const [error, setError] = useState('')
    const [apiStatus, setApiStatus] = useState('Checking')

    const [modulePredictions, setModulePredictions] = useState({})
    const [moduleLoading, setModuleLoading] = useState({})
    const [moduleErrors, setModuleErrors] = useState({})
    const [refreshingAll, setRefreshingAll] = useState(false)
    const [customAircraftHistory, setCustomAircraftHistory] = useState(Array(12).fill(''))
    const [customCargoHistory, setCustomCargoHistory] = useState(Array(24).fill(''))

    const [customPressurePassengers, setCustomPressurePassengers] = useState(Array(12).fill(''))
    const [customPressureAircraft, setCustomPressureAircraft] = useState(Array(12).fill(''))
    const [customPressureCargo, setCustomPressureCargo] = useState(Array(12).fill(''))

    const [customTerminal, setCustomTerminal] = useState('Terminal 1')
    const [customTerminalHistory, setCustomTerminalHistory] = useState(Array(12).fill(''))

    const [customResults, setCustomResults] = useState({})
    const [customLoading, setCustomLoading] = useState({})
    const [customErrors, setCustomErrors] = useState({})


    const runSinglePrediction = async (module, records = monthly) => {
        setModuleLoading(current => ({ ...current, [module]: true }))
        setModuleErrors(current => ({ ...current, [module]: '' }))

        try {
            let result

            if (module === 'passenger') {
                if (records.length < 12) {
                    throw new Error('At least 12 months of passenger history are required.')
                }

                const latestTwelve = records.slice(-12)
                const counts = latestTwelve.map(item => Number(item.passengers))
                const target = nextMonth(latestTwelve[latestTwelve.length - 1].month)

                const response = await axios.post(`${API}/forecast/custom`, {
                    passenger_counts: counts,
                    target_year: target.year,
                    target_month: target.month
                })

                result = response.data
            }

            if (module === 'aircraft') {
                const response = await axios.get(`${API}/forecast/aircraft/latest`)
                result = response.data
            }

            if (module === 'cargo') {
                const response = await axios.get(`${API}/forecast/cargo/latest`)
                result = response.data
            }

            if (module === 'pressure') {
                const response = await axios.get(
                    `${API}/forecast/operational-pressure/ann/latest`
                )
                result = response.data
            }

            if (module === 'terminals') {
                const [terminal1, terminal3] = await Promise.all([
                    axios.get(`${API}/forecast/terminals/Terminal%201`),
                    axios.get(`${API}/forecast/terminals/Terminal%203`)
                ])

                result = {
                    terminal1: terminal1.data,
                    terminal3: terminal3.data
                }
            }

            setModulePredictions(current => ({
                ...current,
                [module]: result
            }))

        } catch (err) {
            console.error(`${module} ANN prediction error:`, err)

            setModuleErrors(current => ({
                ...current,
                [module]: err.response?.data?.detail || err.message || 'Prediction failed.'
            }))
        } finally {
            setModuleLoading(current => ({
                ...current,
                [module]: false
            }))
        }
    }

    const runAllPredictions = async (records = monthly) => {
        setRefreshingAll(true)

        await Promise.all([
            runSinglePrediction('passenger', records),
            runSinglePrediction('aircraft', records),
            runSinglePrediction('cargo', records),
            runSinglePrediction('pressure', records),
            runSinglePrediction('terminals', records)
        ])

        setRefreshingAll(false)
    }

    const loadData = async () => {
        try {
            setLoading(true)
            setError('')
            const response = await axios.get(`${API}/analytics/passengers/detailed`)
            const records = response.data.monthly_traffic || []

            setMonthly(records)
            setApiStatus('Connected')

            if (records.length >= 12) {
                const latestTwelve = records.slice(-12)
                setHistory(latestTwelve.map(item => String(item.passengers)))

                const next = nextMonth(latestTwelve[latestTwelve.length - 1].month)
                setTargetYear(next.year)
                setTargetMonth(next.month)
            }

            runAllPredictions(records)
        } catch (err) {
            console.error('ANN dashboard data error:', err)
            setApiStatus('Offline')
            setError('Unable to load passenger history. Please check that the backend is running.')
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        loadData()
    }, [])

    const renderHistoryInputs = (values, setter, label, placeholder) => (
        <div className="ann-history-grid">
            {values.map((value, index) => (
                <label key={index} className="ann-input-group">
                    <span>{label} {index + 1}</span>
                    <input
                        type="number"
                        min="0"
                        step="any"
                        value={value}
                        onChange={event =>
                            updateCustomHistory(setter, index, event.target.value)
                        }
                        placeholder={placeholder}
                    />
                </label>
            ))}
        </div>
    )

    const latestRecord = monthly[monthly.length - 1]

    const totalPassengers = useMemo(
        () => monthly.reduce((sum, item) => sum + Number(item.passengers || 0), 0),
        [monthly]
    )

    const chartData = useMemo(
        () => monthly.slice(-48).map(item => ({
            month: item.month,
            passengers: Number(item.passengers || 0)
        })),
        [monthly]
    )

    const averagePassengers = monthly.length
        ? totalPassengers / monthly.length
        : 0

    const runPrediction = async event => {
        event.preventDefault()
        setError('')
        setPrediction(null)

        const counts = history.map(value => Number(value))

        if (history.some(value => value === '' || !Number.isFinite(Number(value)))) {
            setError('Please enter all 12 monthly passenger counts.')
            return
        }

        if (counts.some(value => value < 0)) {
            setError('Passenger counts cannot be negative.')
            return
        }

        try {
            setPredicting(true)

            const response = await axios.post(`${API}/forecast/custom`, {
                passenger_counts: counts,
                target_year: Number(targetYear),
                target_month: Number(targetMonth)
            })

            setPrediction(response.data)
        } catch (err) {
            console.error('ANN prediction error:', err)
            setError(err.response?.data?.detail || 'Prediction failed. Please check the backend and model files.')
        } finally {
            setPredicting(false)
        }
    }

    const updateHistory = (index, value) => {
        setHistory(current =>
            current.map((item, i) => i === index ? value : item)
        )
    }

    const updateCustomHistory = (setter, index, value) => {
        setter(current =>
            current.map((item, i) => i === index ? value : item)
        )
    }

    const runCustomPrediction = async (module, endpoint, payload) => {
        setCustomErrors(current => ({ ...current, [module]: '' }))
        setCustomResults(current => ({ ...current, [module]: null }))

        try {
            setCustomLoading(current => ({ ...current, [module]: true }))

            const response = await axios.post(`${API}${endpoint}`, payload)

            setCustomResults(current => ({
                ...current,
                [module]: response.data
            }))
        } catch (err) {
            console.error(`${module} custom prediction error:`, err)

            setCustomErrors(current => ({
                ...current,
                [module]: err.response?.data?.detail || 'Custom prediction failed. Check the backend.'
            }))
        } finally {
            setCustomLoading(current => ({ ...current, [module]: false }))
        }
    }

    const validateHistory = (values, expectedLength, label) => {
        if (values.length !== expectedLength || values.some(value => value === '' || !Number.isFinite(Number(value)))) {
            return `Please enter all ${expectedLength} ${label} values.`
        }

        if (values.some(value => Number(value) < 0)) {
            return `${label} values cannot be negative.`
        }

        return ''
    }

    const runAircraftCustomPrediction = () => {
        const validation = validateHistory(customAircraftHistory, 12, 'monthly aircraft landing')

        if (validation) {
            setCustomErrors(current => ({ ...current, aircraft: validation }))
            return
        }

        runCustomPrediction(
            'aircraft',
            '/forecast/aircraft/custom-history',
            {
                landing_counts: customAircraftHistory.map(Number),
                target_year: Number(targetYear),
                target_month: Number(targetMonth)
            }
        )
    }

    const runCargoCustomPrediction = () => {
        const validation = validateHistory(customCargoHistory, 24, 'monthly cargo weight')

        if (validation) {
            setCustomErrors(current => ({ ...current, cargo: validation }))
            return
        }

        runCustomPrediction(
            'cargo',
            '/forecast/cargo/custom-history',
            {
                cargo_weights: customCargoHistory.map(Number),
                target_year: Number(targetYear),
                target_month: Number(targetMonth)
            }
        )
    }

    const runPressureCustomPrediction = () => {
        const validation =
            validateHistory(customPressurePassengers, 12, 'monthly passenger') ||
            validateHistory(customPressureAircraft, 12, 'monthly aircraft landing') ||
            validateHistory(customPressureCargo, 12, 'monthly cargo weight')

        if (validation) {
            setCustomErrors(current => ({ ...current, pressure: validation }))
            return
        }

        runCustomPrediction(
            'pressure',
            '/forecast/operational-pressure/ann/custom-history',
            {
                passenger_counts: customPressurePassengers.map(Number),
                landing_counts: customPressureAircraft.map(Number),
                cargo_weights: customPressureCargo.map(Number),
                target_year: Number(targetYear),
                target_month: Number(targetMonth)
            }
        )
    }

    const runTerminalCustomPrediction = () => {
        const validation = validateHistory(customTerminalHistory, 12, 'monthly terminal passenger')

        if (validation) {
            setCustomErrors(current => ({ ...current, terminal: validation }))
            return
        }

        runCustomPrediction(
            'terminal',
            `/forecast/terminals/ann/custom-history/${encodeURIComponent(customTerminal)}`,
            {
                passenger_counts: customTerminalHistory.map(Number),
                target_year: Number(targetYear),
                target_month: Number(targetMonth)
            }
        )
    }


    const latestMonth = latestRecord?.month || 'N/A'

    return (
        <div className="ann-page">
            <div className="ann-scene">
                <AirportScene />
            </div>
            <div className="ann-scene-overlay" />

            <div className="ann-content">
                <motion.div
                    className="ann-heading"
                    initial={{ opacity: 0, y: 18 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.7 }}
                >
                    <div>
                        <div className="ann-eyebrow">
                            <BrainCircuit size={16} />
                            AEROCAST AI / INTELLIGENCE SYSTEM
                        </div>
                        <h1>ANN <span>Forecasting</span></h1>
                        <p>
                            Neural network-powered passenger volume prediction for
                            San Francisco International Airport.
                        </p>
                    </div>

                    <div className="ann-live-status">
                        <span className={`ann-status-dot ${apiStatus === 'Connected' ? 'online' : ''}`} />
                        SYSTEM {apiStatus.toUpperCase()}
                    </div>
                </motion.div>

                <div className="ann-hero">
                    <div className="ann-hero-copy">
                        <span className="ann-hero-label">
                            <Sparkles size={15} /> AI PASSENGER INTELLIGENCE
                        </span>
                        <h2>Predict the flow.<br /><span>Prepare the future.</span></h2>
                        <p>
                            Analyze historical passenger movement and generate
                            forecasts using the trained Artificial Neural Network.
                        </p>
                        <div className="ann-hero-tags">
                            <span><BrainCircuit size={14} /> ANN MODEL</span>
                            <span><Database size={14} /> SFO DATA</span>
                            <span><Activity size={14} /> MONTHLY FORECAST</span>
                        </div>
                    </div>
                    <div className="ann-hero-visual">
                        <div className="ann-hero-orbit orbit-one" />
                        <div className="ann-hero-orbit orbit-two" />
                        <div className="ann-hero-plane"><Plane size={78} /></div>
                        <div className="ann-hero-coordinates">37.6213° N<br />122.3790° W</div>
                    </div>
                </div>

                {loading ? (
                    <div className="ann-loading">
                        <RefreshCw className="ann-spin" size={24} />
                        Loading ANN forecasting data...
                    </div>
                ) : (
                    <>
                        <div className="ann-kpi-grid">
                            <motion.div className="ann-kpi-card" whileHover={{ y: -5 }}>
                                <div className="ann-kpi-icon"><Users size={21} /></div>
                                <span>Latest Actual Passengers</span>
                                <h2>{compact(latestRecord?.passengers || 0)}</h2>
                                <p><CalendarDays size={13} /> {latestMonth}</p>
                            </motion.div>

                            <motion.div className="ann-kpi-card" whileHover={{ y: -5 }}>
                                <div className="ann-kpi-icon"><BrainCircuit size={21} /></div>
                                <span>ANN Predicted Passengers</span>
                                <h2>{prediction ? compact(prediction.predicted_passengers) : '—'}</h2>
                                <p><Sparkles size={13} /> {prediction ? prediction.target_month : 'Run a forecast'}</p>
                            </motion.div>

                            <motion.div className="ann-kpi-card" whileHover={{ y: -5 }}>
                                <div className="ann-kpi-icon"><Activity size={21} /></div>
                                <span>Historical Monthly Average</span>
                                <h2>{compact(averagePassengers)}</h2>
                                <p><Database size={13} /> {number(monthly.length)} monthly records</p>
                            </motion.div>

                            <motion.div className="ann-kpi-card" whileHover={{ y: -5 }}>
                                <div className="ann-kpi-icon"><Layers3 size={21} /></div>
                                <span>Forecast Model</span>
                                <h2 className="ann-model-name">ANN</h2>
                                <p><CircleCheck size={13} /> Trained model loaded by API</p>
                            </motion.div>
                        </div>

                        <div className="ann-section-heading">
                            <div>
                                <span>01 / HISTORICAL DATA</span>
                                <h2>Passenger Traffic Intelligence</h2>
                                <p>Historical monthly passenger volume from the SFO dataset</p>
                            </div>
                            <button className="ann-refresh" onClick={loadData}>
                                <RefreshCw size={15} /> Refresh Data
                            </button>
                        </div>

                        <div className="ann-chart-grid">
                            <div className="ann-glass-card ann-chart-main">
                                <div className="ann-card-heading">
                                    <div>
                                        <h3>Passenger Traffic History</h3>
                                        <p>Last 48 available months</p>
                                    </div>
                                    <span className="ann-chart-badge">ACTUAL DATA</span>
                                </div>
                                <ResponsiveContainer width="100%" height={290}>
                                    <AreaChart data={chartData}>
                                        <defs>
                                            <linearGradient id="annPassengerGradient" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#00E6A0" stopOpacity={0.38} />
                                                <stop offset="95%" stopColor="#00E6A0" stopOpacity={0.01} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#285346" />
                                        <XAxis
                                            dataKey="month"
                                            tickFormatter={monthName}
                                            minTickGap={35}
                                            stroke="#8BFFD5"
                                            tick={{ fontSize: 10 }}
                                        />
                                        <YAxis
                                            tickFormatter={compact}
                                            stroke="#8BFFD5"
                                            tick={{ fontSize: 10 }}
                                        />
                                        <Tooltip
                                            labelFormatter={monthName}
                                            formatter={value => [number(value), 'Passengers']}
                                            contentStyle={{
                                                background: '#10352B',
                                                border: '1px solid #00E6A0',
                                                borderRadius: 10,
                                                color: '#EAF4FF'
                                            }}
                                        />
                                        <Area
                                            type="monotone"
                                            dataKey="passengers"
                                            stroke="#00E6A0"
                                            strokeWidth={2.5}
                                            fill="url(#annPassengerGradient)"
                                        />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>

                            <div className="ann-glass-card ann-model-card">
                                <div className="ann-card-heading">
                                    <div>
                                        <h3>Neural Network</h3>
                                        <p>Prediction pipeline</p>
                                    </div>
                                    <BrainCircuit size={22} />
                                </div>

                                <div className="ann-network">
                                    <div className="ann-network-node input-node">
                                        <Database size={19} />
                                        <strong>12</strong>
                                        <span>Input features</span>
                                    </div>
                                    <div className="ann-network-lines">••••••••••••</div>
                                    <div className="ann-network-node hidden-node">
                                        <Cpu size={19} />
                                        <strong>ANN</strong>
                                        <span>Hidden layers</span>
                                    </div>
                                    <div className="ann-network-lines">••••••••••••</div>
                                    <div className="ann-network-node output-node">
                                        <Gauge size={19} />
                                        <strong>1</strong>
                                        <span>Prediction</span>
                                    </div>
                                </div>

                                <div className="ann-model-info">
                                    <div><span>Model type</span><strong>Artificial Neural Network</strong></div>
                                    <div><span>Input features</span><strong>12</strong></div>
                                    <div><span>Output</span><strong>Monthly passenger count</strong></div>
                                    <div><span>Model file</span><strong>ann_passenger_model.keras</strong></div>
                                </div>
                                <p className="ann-model-note">
                                    Architecture illustration. Exact hidden-layer sizes are not
                                    displayed here because they have not yet been inspected.
                                </p>
                            </div>
                        </div>

                        <div className="ann-section-heading">
                            <div>
                                <span>02 / PREDICTION ENGINE</span>
                                <h2>Custom Passenger Forecast</h2>
                                <p>Provide 12 consecutive monthly passenger counts to predict a target month</p>
                            </div>
                        </div>

                        <div className="ann-forecast-layout">
                            <form className="ann-glass-card ann-forecast-form" onSubmit={runPrediction}>
                                <div className="ann-card-heading">
                                    <div>
                                        <h3>Historical Input Data</h3>
                                        <p>Enter values from oldest month to newest month</p>
                                    </div>
                                    <BrainCircuit size={23} />
                                </div>

                                <div className="ann-history-grid">
                                    {history.map((value, index) => (
                                        <label key={index} className="ann-input-group">
                                            <span>Month {index + 1}</span>
                                            <input
                                                type="number"
                                                min="0"
                                                value={value}
                                                onChange={event => updateHistory(index, event.target.value)}
                                                placeholder="Passenger count"
                                                required
                                            />
                                        </label>
                                    ))}
                                </div>

                                <div className="ann-target-row">
                                    <label>
                                        <span>Target Year</span>
                                        <input
                                            type="number"
                                            min="1999"
                                            max="2100"
                                            value={targetYear}
                                            onChange={event => setTargetYear(event.target.value)}
                                            required
                                        />
                                    </label>
                                    <label>
                                        <span>Target Month</span>
                                        <select
                                            value={targetMonth}
                                            onChange={event => setTargetMonth(Number(event.target.value))}
                                        >
                                            {Array.from({ length: 12 }).map((_, index) => (
                                                <option key={index} value={index + 1}>
                                                    {new Date(2000, index).toLocaleDateString('en-US', { month: 'long' })}
                                                </option>
                                            ))}
                                        </select>
                                    </label>
                                </div>

                                {error && (
                                    <div className="ann-error">
                                        <AlertCircle size={16} /> {error}
                                    </div>
                                )}

                                <button className="ann-predict-button" type="submit" disabled={predicting}>
                                    {predicting ? (
                                        <><RefreshCw className="ann-spin" size={17} /> Generating Prediction...</>
                                    ) : (
                                        <><Send size={17} /> Generate ANN Prediction</>
                                    )}
                                </button>
                            </form>

                            <div className="ann-glass-card ann-result-card">
                                <div className="ann-card-heading">
                                    <div>
                                        <h3>Prediction Result</h3>
                                        <p>ANN model output</p>
                                    </div>
                                    <Sparkles size={22} />
                                </div>

                                {prediction ? (
                                    <motion.div
                                        className="ann-result-success"
                                        initial={{ opacity: 0, scale: 0.92 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                    >
                                        <span className="ann-result-label">PREDICTED PASSENGER VOLUME</span>
                                        <h2>{number(prediction.predicted_passengers)}</h2>
                                        <p>Passengers · {monthName(prediction.target_month)}</p>
                                        <div className="ann-result-status">
                                            <CircleCheck size={16} /> Prediction generated successfully
                                        </div>
                                        <div className="ann-result-detail">
                                            <span>Model</span>
                                            <strong>{prediction.model_name}</strong>
                                        </div>
                                        <div className="ann-result-detail">
                                            <span>Input months</span>
                                            <strong>{prediction.input_history?.length || 12}</strong>
                                        </div>
                                    </motion.div>
                                ) : (
                                    <div className="ann-result-empty">
                                        <div className="ann-result-orb"><BrainCircuit size={42} /></div>
                                        <h4>Awaiting Prediction</h4>
                                        <p>
                                            Enter your 12-month passenger history and generate
                                            a forecast to see the ANN model output here.
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>


                        <div className="ann-section-heading ann-modules-heading">
                            <div>
                                <span>03 / MULTI-MODEL PREDICTION ENGINE</span>
                                <h2>Airport-Wide ANN Predictions</h2>
                                <p>Five prediction categories powered by your trained neural network models</p>
                            </div>

                            <button
                                className="ann-refresh"
                                onClick={() => runAllPredictions()}
                                disabled={refreshingAll}
                            >
                                <RefreshCw size={15} className={refreshingAll ? 'ann-spin' : ''} />
                                {refreshingAll ? 'Refreshing...' : 'Refresh All Predictions'}
                            </button>
                        </div>

                        <div className="ann-prediction-grid">

                            <div className="ann-glass-card ann-module-card">
                                <div className="ann-card-heading">
                                    <div>
                                        <h3><Users size={18} /> Passenger Volume</h3>
                                        <p>Monthly passenger forecast</p>
                                    </div>
                                    <BrainCircuit size={22} />
                                </div>

                                <div className="ann-module-value">
                                    {modulePredictions.passenger
                                        ? number(modulePredictions.passenger.predicted_passengers)
                                        : '—'}
                                </div>
                                <span className="ann-module-unit">Passengers</span>
                                <div className="ann-module-meta">
                                    {modulePredictions.passenger
                                        ? `Target: ${monthName(modulePredictions.passenger.target_month)}`
                                        : moduleErrors.passenger || 'Awaiting prediction'}
                                </div>

                                <div className="ann-custom-panel">
                                    <h4>Custom Passenger Prediction</h4>
                                    <p>Enter your own passenger history to generate a forecast.</p>

                                    <h5>Passenger History (Last 12 Months)</h5>

                                    <div className="ann-history-grid">
                                        {history.map((value, index) => (
                                            <label key={index} className="ann-input-group">
                                                <span>Month {index + 1}</span>
                                                <input
                                                    type="number"
                                                    min="0"
                                                    step="any"
                                                    value={value}
                                                    onChange={event => updateHistory(index, event.target.value)}
                                                    placeholder="Enter passenger count"
                                                />
                                            </label>
                                        ))}
                                    </div>

                                    <button
                                        type="button"
                                        className="ann-predict-button"
                                        onClick={runPrediction}
                                        disabled={predicting}
                                    >
                                        {predicting ? 'Predicting...' : 'Predict Custom Passenger'}
                                    </button>

                                    {error && <p className="ann-error">{error}</p>}

                                    {prediction && (
                                        <div className="ann-custom-result">
                                            <span>CUSTOM PASSENGER FORECAST</span>
                                            <strong>{number(prediction.predicted_passengers)}</strong>
                                            <p>Target: {monthName(prediction.target_month)}</p>
                                        </div>
                                    )}
                                </div>

                                <button
                                    className="ann-module-button"
                                    onClick={() => runSinglePrediction('passenger')}
                                    disabled={moduleLoading.passenger || monthly.length < 12}
                                >
                                    {moduleLoading.passenger ? 'Predicting...' : 'Run Prediction'}
                                </button>
                            </div>

                            <div className="ann-glass-card ann-module-card">
                                <div className="ann-card-heading">
                                    <div>
                                        <h3><PlaneTakeoff size={18} /> Aircraft Movement</h3>
                                        <p>Monthly aircraft landings</p>
                                    </div>
                                    <Activity size={22} />
                                </div>
                                <div className="ann-module-value">
                                    {modulePredictions.aircraft
                                        ? number(modulePredictions.aircraft.predicted_landings)
                                        : '—'}
                                </div>
                                <span className="ann-module-unit">Predicted landings</span>
                                <div className="ann-module-meta">
                                    {modulePredictions.aircraft?.input_month
                                        ? `Data month: ${modulePredictions.aircraft.input_month}`
                                        : moduleErrors.aircraft || 'Awaiting prediction'}
                                </div>

                                <div className="ann-custom-panel">
                                    <h4>Custom Aircraft Prediction</h4>
                                    <p>Enter 12 monthly landing counts, oldest to newest.</p>

                                    {renderHistoryInputs(
                                        customAircraftHistory,
                                        setCustomAircraftHistory,
                                        'Month',
                                        'Landing count'
                                    )}

                                    {customErrors.aircraft && (
                                        <div className="ann-error">
                                            <AlertCircle size={16} />
                                            {customErrors.aircraft}
                                        </div>
                                    )}

                                    <button
                                        type="button"
                                        className="ann-predict-button"
                                        onClick={runAircraftCustomPrediction}
                                        disabled={customLoading.aircraft}
                                    >
                                        {customLoading.aircraft ? 'Predicting...' : 'Predict Custom Aircraft'}
                                    </button>

                                    {customResults.aircraft && (
                                        <div className="ann-custom-result">
                                            <span>YOUR CUSTOM AIRCRAFT FORECAST</span>
                                            <strong>
                                                {number(customResults.aircraft.predicted_landings)}
                                            </strong>
                                            <p>Predicted aircraft landings</p>
                                        </div>
                                    )}
                                </div>

                                <button
                                    className="ann-module-button"
                                    onClick={() => runSinglePrediction('aircraft')}
                                    disabled={moduleLoading.aircraft}
                                >
                                    {moduleLoading.aircraft ? 'Predicting...' : 'Run Prediction'}
                                </button>
                            </div>

                            <div className="ann-glass-card ann-module-card">
                                <div className="ann-card-heading">
                                    <div>
                                        <h3><Database size={18} /> Cargo Volume</h3>
                                        <p>Monthly cargo weight</p>
                                    </div>
                                    <Layers3 size={22} />
                                </div>
                                <div className="ann-module-value">
                                    {modulePredictions.cargo
                                        ? compact(modulePredictions.cargo.predicted_cargo_weight_lbs)
                                        : '—'}
                                </div>
                                <span className="ann-module-unit">Pounds (lbs)</span>
                                <div className="ann-module-meta">
                                    {modulePredictions.cargo?.input_month
                                        ? `Data month: ${modulePredictions.cargo.input_month}`
                                        : moduleErrors.cargo || 'Awaiting prediction'}
                                </div>

                                <div className="ann-custom-panel">
                                    <h4>Custom Cargo Prediction</h4>
                                    <p>Enter 24 monthly cargo weights, oldest to newest.</p>

                                    {renderHistoryInputs(
                                        customCargoHistory,
                                        setCustomCargoHistory,
                                        'Month',
                                        'Cargo weight (lbs)'
                                    )}

                                    {customErrors.cargo && (
                                        <div className="ann-error">
                                            <AlertCircle size={16} />
                                            {customErrors.cargo}
                                        </div>
                                    )}

                                    <button
                                        type="button"
                                        className="ann-predict-button"
                                        onClick={runCargoCustomPrediction}
                                        disabled={customLoading.cargo}
                                    >
                                        {customLoading.cargo ? 'Predicting...' : 'Predict Custom Cargo'}
                                    </button>

                                    {customResults.cargo && (
                                        <div className="ann-custom-result">
                                            <span>YOUR CUSTOM CARGO FORECAST</span>
                                            <strong>
                                                {number(
                                                    customResults.cargo.predicted_cargo_weight_lbs ??
                                                    customResults.cargo.predicted_cargo_weight ??
                                                    customResults.cargo.predicted_weight ??
                                                    customResults.cargo.prediction
                                                )}
                                            </strong>
                                            <p>Predicted cargo weight (lbs)</p>
                                        </div>
                                    )}
                                </div>

                                <button
                                    className="ann-module-button"
                                    onClick={() => runSinglePrediction('cargo')}
                                    disabled={moduleLoading.cargo}
                                >
                                    {moduleLoading.cargo ? 'Predicting...' : 'Run Prediction'}
                                </button>
                            </div>

                            <div className="ann-glass-card ann-module-card">
                                <div className="ann-card-heading">
                                    <div>
                                        <h3><Gauge size={18} /> Operational Pressure</h3>
                                        <p>Airport pressure index</p>
                                    </div>
                                    <Activity size={22} />
                                </div>
                                <div className="ann-module-value">
                                    {modulePredictions.pressure
                                        ? Number(modulePredictions.pressure.predicted_operational_pressure_index).toFixed(2)
                                        : '—'}
                                </div>
                                <span className="ann-module-unit">Predicted pressure index</span>
                                <div className="ann-module-meta">
                                    {modulePredictions.pressure?.input_month
                                        ? `Data month: ${modulePredictions.pressure.input_month}`
                                        : moduleErrors.pressure || 'Awaiting prediction'}
                                </div>

                                <div className="ann-custom-panel">
                                    <h4>Custom Operational Pressure Prediction</h4>
                                    <p>Enter the last 12 monthly values for all three indicators, oldest to newest.</p>

                                    <h5>Passenger Counts</h5>
                                    {renderHistoryInputs(
                                        customPressurePassengers,
                                        setCustomPressurePassengers,
                                        'Month',
                                        'Passenger count'
                                    )}

                                    <h5>Aircraft Landings</h5>
                                    {renderHistoryInputs(
                                        customPressureAircraft,
                                        setCustomPressureAircraft,
                                        'Month',
                                        'Landing count'
                                    )}

                                    <h5>Cargo Weights</h5>
                                    {renderHistoryInputs(
                                        customPressureCargo,
                                        setCustomPressureCargo,
                                        'Month',
                                        'Cargo weight (lbs)'
                                    )}

                                    {customErrors.pressure && (
                                        <div className="ann-error">
                                            <AlertCircle size={16} />
                                            {customErrors.pressure}
                                        </div>
                                    )}

                                    <button
                                        type="button"
                                        className="ann-predict-button"
                                        onClick={runPressureCustomPrediction}
                                        disabled={customLoading.pressure}
                                    >
                                        {customLoading.pressure ? 'Predicting...' : 'Predict Custom Pressure'}
                                    </button>

                                    {customResults.pressure && (
                                        <div className="ann-custom-result">
                                            <span>YOUR CUSTOM PRESSURE FORECAST</span>
                                            <strong>
                                                {number(
                                                    customResults.pressure.predicted_operational_pressure_index ??
                                                    customResults.pressure.predicted_pressure_index ??
                                                    customResults.pressure.prediction
                                                )}
                                            </strong>
                                            <p>Predicted operational pressure index</p>
                                        </div>
                                    )}
                                </div>


                                <button
                                    className="ann-module-button"
                                    onClick={() => runSinglePrediction('pressure')}
                                    disabled={moduleLoading.pressure}
                                >
                                    {moduleLoading.pressure ? 'Predicting...' : 'Run Prediction'}
                                </button>
                            </div>

                            <div className="ann-glass-card ann-module-card ann-terminal-module">
                                <div className="ann-card-heading">
                                    <div>
                                        <h3><Layers3 size={18} /> Terminal-wise Passengers</h3>
                                        <p>Individual terminal predictions</p>
                                    </div>
                                    <Plane size={22} />
                                </div>

                                <div className="ann-terminal-results">
                                    <div>
                                        <span>TERMINAL 1</span>
                                        <strong>
                                            {modulePredictions.terminals
                                                ? number(modulePredictions.terminals.terminal1.predicted_passengers)
                                                : '—'}
                                        </strong>
                                        <small>Passengers</small>
                                    </div>
                                    <div>
                                        <span>TERMINAL 3</span>
                                        <strong>
                                            {modulePredictions.terminals
                                                ? number(modulePredictions.terminals.terminal3.predicted_passengers)
                                                : '—'}
                                        </strong>
                                        <small>Passengers</small>
                                    </div>
                                </div>

                                <div className="ann-custom-panel">
                                    <h4>Custom Terminal Passenger Prediction</h4>
                                    <p>Choose a terminal and enter 12 monthly passenger counts, oldest to newest.</p>

                                    <label className="ann-input-group">
                                        <span>Select Terminal</span>
                                        <select
                                            value={customTerminal}
                                            onChange={event => setCustomTerminal(event.target.value)}
                                        >
                                            <option value="Terminal 1">Terminal 1</option>
                                            <option value="Terminal 3">Terminal 3</option>
                                        </select>
                                    </label>

                                    {renderHistoryInputs(
                                        customTerminalHistory,
                                        setCustomTerminalHistory,
                                        'Month',
                                        'Passenger count'
                                    )}

                                    {customErrors.terminal && (
                                        <div className="ann-error">
                                            <AlertCircle size={16} />
                                            {customErrors.terminal}
                                        </div>
                                    )}

                                    <button
                                        type="button"
                                        className="ann-predict-button"
                                        onClick={runTerminalCustomPrediction}
                                        disabled={customLoading.terminal}
                                    >
                                        {customLoading.terminal ? 'Predicting...' : 'Predict Custom Terminal'}
                                    </button>

                                    {customResults.terminal && (
                                        <div className="ann-custom-result">
                                            <span>{customTerminal.toUpperCase()} CUSTOM FORECAST</span>
                                            <strong>
                                                {number(customResults.terminal.predicted_passengers)}
                                            </strong>
                                            <p>Predicted terminal passengers</p>
                                        </div>
                                    )}
                                </div>


                                <div className="ann-module-meta">
                                    {modulePredictions.terminals
                                        ? `Data month: ${modulePredictions.terminals.terminal1.input_month}`
                                        : moduleErrors.terminals || 'Awaiting prediction'}
                                </div>
                                <button
                                    className="ann-module-button"
                                    onClick={() => runSinglePrediction('terminals')}
                                    disabled={moduleLoading.terminals}
                                >
                                    {moduleLoading.terminals ? 'Predicting...' : 'Run Prediction'}
                                </button>
                            </div>

                        </div>

                        <div className="ann-bottom-note">
                            <PlaneTakeoff size={18} />
                            <div>
                                <strong>AeroCast AI · SFO Passenger Forecasting</strong>
                                <p>
                                    Predictions are generated by the trained ANN model using
                                    the supplied 12-month passenger history.
                                </p>
                            </div>
                            <ArrowUpRight size={18} />
                        </div>
                    </>
                )}
            </div>
        </div>
    )
}

export default ANNForecasting