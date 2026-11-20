
import axios from 'axios'

const API_BASE_URL = 'http://127.0.0.1:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const getPassengerAnalytics = () =>
  api.get('/api/analytics/passengers/detailed')

export const getAircraftAnalytics = () =>
  api.get('/api/analytics/aircraft/monthly')

export const getCargoAnalytics = () =>
  api.get('/api/analytics/cargo/monthly')

export const getIntegratedAnalytics = () =>
  api.get('/api/analytics/integrated')

export const getForecast = (data) =>
  api.post('/api/forecast', data)

// Airport Live View API
export const getLiveAircraft = () =>
  api.get('/api/airport/live')

export default api
