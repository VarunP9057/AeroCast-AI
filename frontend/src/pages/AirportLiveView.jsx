import twinsImage from '../assets/images/twins.png'
import { useCallback, useEffect, useMemo, useState } from 'react'
import {
    MapContainer,
    TileLayer,
    Marker,
    Popup,
    Circle,
    useMap
} from 'react-leaflet'
import L from 'leaflet'
import {
    Radio,
    Plane,
    PlaneLanding,
    PlaneTakeoff,
    RefreshCw,
    MapPin,
    Activity,
    Gauge,
    Navigation,
    Clock,
    Wifi,
    AlertCircle
} from 'lucide-react'

import { getLiveAircraft } from '../services/api'
import 'leaflet/dist/leaflet.css'
import './AirportLiveView.css'

const SFO_POSITION = [37.6213, -122.3790]

function createAircraftIcon(heading = 0, onGround = false) {
    const rotation = Number(heading) || 0

    return L.divIcon({
        className: 'aircraft-marker-wrapper',
        html: `
      <div class="aircraft-marker ${onGround ? 'aircraft-ground' : ''}"
           style="transform: rotate(${rotation}deg)">
        <span>✈</span>
      </div>
    `,
        iconSize: [36, 36],
        iconAnchor: [18, 18]
    })
}

function MapResizeHandler() {
    const map = useMap()

    useEffect(() => {
        const timer = setTimeout(() => map.invalidateSize(), 200)
        return () => clearTimeout(timer)
    }, [map])

    return null
}

function formatAltitude(value) {
    if (value === null || value === undefined) return 'N/A'
    if (value === 'ground') return 'Ground'
    return `${Number(value).toLocaleString()} ft`
}

function formatSpeed(value) {
    if (value === null || value === undefined) return 'N/A'
    return `${Math.round(Number(value))} kts`
}

function formatTime(value) {
    if (!value) return 'Waiting for data'

    return new Date(value).toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    })
}

export default function AirportLiveView() {
    const [aircraft, setAircraft] = useState([])
    const [loading, setLoading] = useState(true)
    const [refreshing, setRefreshing] = useState(false)
    const [error, setError] = useState('')
    const [lastUpdated, setLastUpdated] = useState(null)
    const [airportName, setAirportName] = useState(
        'San Francisco International Airport'
    )

    const fetchAircraft = useCallback(async (manual = false) => {
        if (manual) {
            setRefreshing(true)
        } else {
            setLoading(true)
        }

        try {
            const response = await getLiveAircraft()
            const data = response.data

            setAircraft(Array.isArray(data.aircraft) ? data.aircraft : [])
            setAirportName(data.airport || 'San Francisco International Airport')
            setLastUpdated(data.fetched_at || new Date().toISOString())
            setError('')
        } catch (err) {
            setError(
                err.response?.data?.detail ||
                'Unable to connect to the live aircraft data service.'
            )
        } finally {
            setLoading(false)
            setRefreshing(false)
        }
    }, [])

    useEffect(() => {
        fetchAircraft()

        const interval = setInterval(() => {
            fetchAircraft()
        }, 60000)

        return () => clearInterval(interval)
    }, [fetchAircraft])

    const airborneAircraft = useMemo(
        () => aircraft.filter(item => !item.on_ground),
        [aircraft]
    )

    const groundAircraft = useMemo(
        () => aircraft.filter(item => item.on_ground),
        [aircraft]
    )

    const averageSpeed = useMemo(() => {
        const speeds = airborneAircraft
            .map(item => Number(item.speed))
            .filter(speed => Number.isFinite(speed) && speed > 0)

        if (!speeds.length) return 0

        return Math.round(
            speeds.reduce((total, speed) => total + speed, 0) / speeds.length
        )
    }, [airborneAircraft])

    return (
        <div className="airport-live-page">

            <div className="live-page-header">
                <div>
                    <div className="live-eyebrow">
                        <Radio size={15} />
                        AIRPORT SURVEILLANCE SYSTEM
                    </div>

                    <h1>Airport Live View</h1>

                    <p>
                        Real-time aircraft tracking and airport traffic monitoring
                    </p>
                </div>

                <div className="live-header-actions">
                    <div className="live-connection">
                        <span className="live-pulse"></span>
                        {error ? 'CONNECTION ISSUE' : 'LIVE TRACKING'}
                    </div>

                    <button
                        className="live-refresh-button"
                        onClick={() => fetchAircraft(true)}
                        disabled={refreshing}
                    >
                        <RefreshCw
                            size={16}
                            className={refreshing ? 'refresh-spinning' : ''}
                        />
                        Refresh
                    </button>
                </div>
            </div>

            <div className="airport-info-banner">
                <div className="airport-info-left">
                    <div className="airport-location-icon">
                        <MapPin size={23} />
                    </div>

                    <div>
                        <span className="airport-small-label">MONITORING AIRPORT</span>
                        <h2>{airportName}</h2>
                        <p>San Francisco, California · United States</p>
                    </div>
                </div>

                <div className="airport-code-box">
                    <span>AIRPORT CODE</span>
                    <strong>KSFO</strong>
                </div>
            </div>

            <div className="live-stat-grid">

                <div className="live-stat-card">
                    <div className="live-stat-top">
                        <span>Total Aircraft</span>
                        <div className="live-stat-icon cyan">
                            <Plane size={20} />
                        </div>
                    </div>
                    <strong>{loading ? '--' : aircraft.length}</strong>
                    <small>Detected within 25 NM</small>
                </div>

                <div className="live-stat-card">
                    <div className="live-stat-top">
                        <span>Airborne</span>
                        <div className="live-stat-icon green">
                            <PlaneTakeoff size={20} />
                        </div>
                    </div>
                    <strong>{loading ? '--' : airborneAircraft.length}</strong>
                    <small>Aircraft in the air</small>
                </div>

                <div className="live-stat-card">
                    <div className="live-stat-top">
                        <span>On Ground</span>
                        <div className="live-stat-icon purple">
                            <PlaneLanding size={20} />
                        </div>
                    </div>
                    <strong>{loading ? '--' : groundAircraft.length}</strong>
                    <small>Ground-detected aircraft</small>
                </div>

                <div className="live-stat-card">
                    <div className="live-stat-top">
                        <span>Average Speed</span>
                        <div className="live-stat-icon orange">
                            <Gauge size={20} />
                        </div>
                    </div>
                    <strong>{loading ? '--' : averageSpeed}</strong>
                    <small>Knots · airborne aircraft</small>
                </div>

            </div>

            {error && (
                <div className="airport-error">
                    <AlertCircle size={19} />
                    <span>{error}</span>
                    <button onClick={() => fetchAircraft(true)}>
                        Retry
                    </button>
                </div>
            )}

            <div className="live-content-grid">

                <section className="live-map-panel">

                    <div className="live-panel-header">
                        <div>
                            <h3>Live Aircraft Radar</h3>
                            <p>Aircraft positions around San Francisco International Airport</p>
                        </div>

                        <div className="map-live-label">
                            <span className="live-pulse"></span>
                            LIVE MAP
                        </div>
                    </div>

                    <div className="airport-map-container">

                        <MapContainer
                            center={SFO_POSITION}
                            zoom={11}
                            scrollWheelZoom={true}
                            className="airport-leaflet-map"
                        >
                            <MapResizeHandler />

                            <TileLayer
                                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                            />

                            <Circle
                                center={SFO_POSITION}
                                radius={46300}
                                pathOptions={{
                                    color: '#00d9ff',
                                    fillColor: '#00d9ff',
                                    fillOpacity: 0.035,
                                    weight: 1,
                                    dashArray: '7 8'
                                }}
                            />

                            {aircraft.map((item, index) => (
                                <Marker
                                    key={item.hex || `${item.callsign}-${index}`}
                                    position={[item.latitude, item.longitude]}
                                    icon={createAircraftIcon(
                                        item.heading,
                                        item.on_ground
                                    )}
                                >
                                    <Popup>
                                        <div className="aircraft-popup">
                                            <strong>
                                                {item.callsign || 'Unknown Aircraft'}
                                            </strong>
                                            <span>
                                                Type: {item.aircraft_type || 'Unknown'}
                                            </span>
                                            <span>
                                                Altitude: {formatAltitude(item.altitude)}
                                            </span>
                                            <span>
                                                Speed: {formatSpeed(item.speed)}
                                            </span>
                                            <span>
                                                Heading: {item.heading ?? 'N/A'}°
                                            </span>
                                        </div>
                                    </Popup>
                                </Marker>
                            ))}

                            <Marker position={SFO_POSITION}>
                                <Popup>
                                    <strong>San Francisco International Airport</strong>
                                    <br />
                                    KSFO
                                </Popup>
                            </Marker>

                        </MapContainer>

                        <div className="map-overlay-tag">
                            <MapPin size={13} />
                            SFO AIRSPACE
                        </div>

                    </div>

                    <div className="map-legend">
                        <span>
                            <i className="legend-aircraft"></i>
                            Tracked Aircraft
                        </span>
                        <span>
                            <i className="legend-ground"></i>
                            Ground Aircraft
                        </span>
                        <span>
                            <i className="legend-airport"></i>
                            Airport
                        </span>
                    </div>

                </section>

                <section className="aircraft-list-panel">

                    <div className="aircraft-list-header">
                        <div>
                            <h3>Aircraft Activity</h3>
                            <p>Detected aircraft in SFO airspace</p>
                        </div>
                        <span className="aircraft-total-badge">
                            {aircraft.length}
                        </span>
                    </div>

                    <div className="aircraft-list-scroll">

                        {loading && (
                            <div className="aircraft-empty-state">
                                <RefreshCw size={24} className="refresh-spinning" />
                                <p>Loading aircraft positions...</p>
                            </div>
                        )}

                        {!loading && aircraft.length === 0 && !error && (
                            <div className="aircraft-empty-state">
                                <Plane size={27} />
                                <p>No aircraft currently detected in the returned data.</p>
                            </div>
                        )}

                        {!loading && aircraft.map((item, index) => (
                            <div
                                className="aircraft-list-item"
                                key={item.hex || `${item.callsign}-${index}`}
                            >
                                <div className="aircraft-list-icon">
                                    <Plane size={19} />
                                </div>

                                <div className="aircraft-list-details">
                                    <strong>
                                        {item.callsign || item.registration || 'Unknown Aircraft'}
                                    </strong>
                                    <span>
                                        {item.aircraft_type || 'Aircraft type unavailable'}
                                    </span>
                                    <div className="aircraft-mini-details">
                                        <span>{formatAltitude(item.altitude)}</span>
                                        <span>{formatSpeed(item.speed)}</span>
                                    </div>
                                </div>

                                <span className={`aircraft-status ${item.on_ground ? 'ground' : 'airborne'}`}>
                                    {item.on_ground ? 'GROUND' : 'AIRBORNE'}
                                </span>
                            </div>
                        ))}

                    </div>

                    <div className="aircraft-list-footer">
                        <Wifi size={15} />
                        ADS-B DATA FEED
                    </div>

                </section>

            </div>
            <section className="digital-twin-preview">

                <div className="live-panel-header">
                    <div>
                        <h3>3D Airport Digital Twin</h3>
                        <p>Visual representation of the airport environment</p>
                    </div>
                </div>

                <img
                    src="/images/twins.png"

                    alt="AeroCast AI 3D Airport Digital Twin"
                    style={{
                        display: 'block',
                        width: '100%',
                        minHeight: '300px',
                        objectFit: 'contain',
                        border: '2px solid cyan'
                    }}
                />
                <p>{String(twinsImage)}</p>

            </section>
           

            <div className="live-bottom-info">

                <div>
                    <Activity size={17} />
                    <span>DATA SOURCE: ADSB.lol</span>
                </div>

                <div>
                    <Clock size={17} />
                    <span>LAST UPDATED: {formatTime(lastUpdated)}</span>
                </div>

                <div>
                    <Navigation size={17} />
                    <span>SEARCH RADIUS: 25 NM</span>
                </div>

            </div>

            <p className="live-disclaimer">
                Aircraft positions are based on available ADS-B reports. Coverage,
                update frequency and aircraft visibility may vary. This display
                does not represent an official air traffic control system.
            </p>

        </div>
    )
}
