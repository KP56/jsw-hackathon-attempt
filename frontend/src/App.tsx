import { useEffect, useState } from 'react'
import './App.css'
import MeasurementCard from './components/MeasurementCard'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

interface Segment {
  segment: number
  start_frame: number
  end_frame: number
  min_x: number | null
  max_x: number | null
}

interface Measurement {
  id: number
  timestamp: string
  video_path: string
  num_segments_requested: number
  num_segments_found: number
  segments: Segment[]
}

interface MeasurementsResponse {
  count: number
  measurements: Measurement[]
}

function App() {
  const [measurements, setMeasurements] = useState<Measurement[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastFetch, setLastFetch] = useState<Date>(new Date())
  const [numSegments, setNumSegments] = useState<number>(5)
  const [analyzing, setAnalyzing] = useState(false)
  const [analysisMessage, setAnalysisMessage] = useState<string | null>(null)

  const fetchMeasurements = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await fetch(`${API_URL}/measurements`)
      
      if (!response.ok) {
        throw new Error('Failed to fetch measurements')
      }
      
      const data: MeasurementsResponse = await response.json()
      setMeasurements(data.measurements)
      setLastFetch(new Date())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const runVideoAnalysis = async () => {
    try {
      setAnalyzing(true)
      setAnalysisMessage('🎬 Video analysis started! Processing in background...')
      setError(null)
      
      const response = await fetch(`${API_URL}/analyze_video?num_segments=${numSegments}`)
      
      if (!response.ok) {
        throw new Error('Failed to start video analysis')
      }
      
      const data = await response.json()
      
      // Check if it's the new async response (202) or old sync response (200)
      if (response.status === 202 || data.status === 'accepted') {
        setAnalysisMessage(`✓ Analysis started! Processing ${numSegments} segments in background. Results will appear below when complete.`)
        
        // Poll for new measurements every 3 seconds
        let pollCount = 0
        const maxPolls = 100 // Maximum 5 minutes (100 * 3 seconds)
        
        const pollInterval = setInterval(() => {
          pollCount++
          fetchMeasurements()
          
          if (pollCount >= maxPolls) {
            clearInterval(pollInterval)
            setAnalysisMessage('⏱️ Analysis is taking longer than expected. Check backend logs for progress.')
            setAnalyzing(false)
          }
        }, 3000)
        
        // Clear message after 10 seconds
        setTimeout(() => {
          setAnalysisMessage(null)
          clearInterval(pollInterval)
          setAnalyzing(false)
        }, 10000)
      } else {
        // Old sync response
        setAnalysisMessage(`✓ Analysis complete! Found ${data.num_segments_found} segments.`)
        
        setTimeout(() => {
          fetchMeasurements()
          setAnalysisMessage(null)
        }, 2000)
        setAnalyzing(false)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Video analysis failed')
      setAnalysisMessage(null)
      setAnalyzing(false)
    }
  }

  useEffect(() => {
    fetchMeasurements()
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchMeasurements, 30000)
    
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="app">
      <header className="header">
        <h1>Video Segmentation Measurements</h1>
        <p className="subtitle">Real-time monitoring of the 10 most recent measurements</p>
        
        <div className="analysis-section">
          <h3>Run Video Analysis</h3>
          <div className="analysis-controls">
            <div className="input-group">
              <label htmlFor="num-segments">Number of Segments:</label>
              <input
                id="num-segments"
                type="number"
                min="1"
                max="50"
                value={numSegments}
                onChange={(e) => setNumSegments(parseInt(e.target.value) || 1)}
                disabled={analyzing}
                className="segments-input"
              />
            </div>
            <button 
              onClick={runVideoAnalysis} 
              className="analyze-btn" 
              disabled={analyzing || numSegments < 1}
            >
              {analyzing ? '⏳ Analyzing...' : '▶️ Analyze Video'}
            </button>
          </div>
          {analysisMessage && (
            <div className={`analysis-message ${analyzing ? 'analyzing' : 'success'}`}>
              {analysisMessage}
            </div>
          )}
        </div>

        <div className="header-actions">
          <button onClick={fetchMeasurements} className="refresh-btn" disabled={loading}>
            {loading ? 'Loading...' : '🔄 Refresh'}
          </button>
          <span className="last-updated">
            Last updated: {lastFetch.toLocaleTimeString()}
          </span>
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <span>⚠️ {error}</span>
          <button onClick={fetchMeasurements}>Try Again</button>
        </div>
      )}

      {!loading && measurements.length === 0 && !error && (
        <div className="empty-state">
          <div className="empty-icon">📊</div>
          <h2>No measurements yet</h2>
          <p>Run a video analysis to see measurements appear here</p>
        </div>
      )}

      <div className="measurements-grid">
        {measurements.map((measurement) => (
          <MeasurementCard key={measurement.id} measurement={measurement} />
        ))}
      </div>
    </div>
  )
}

export default App

