import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import './MeasurementCard.css'

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

interface Props {
  measurement: Measurement
}

function MeasurementCard({ measurement }: Props) {
  // Transform segment data for the chart
  const chartData = measurement.segments.map(segment => ({
    segment: segment.segment,
    min_x: segment.min_x ?? 0,
    max_x: segment.max_x ?? 0,
  }))

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp)
    return date.toLocaleString()
  }

  const getFileName = (path: string) => {
    return path.split(/[\\/]/).pop() || path
  }

  return (
    <div className="measurement-card">
      <div className="card-header">
        <div className="card-title">
          <span className="measurement-id">#{measurement.id}</span>
          <span className="video-name">{getFileName(measurement.video_path)}</span>
        </div>
        <div className="timestamp">{formatTimestamp(measurement.timestamp)}</div>
      </div>

      <div className="card-stats">
        <div className="stat">
          <span className="stat-label">Segments Found</span>
          <span className="stat-value">{measurement.num_segments_found}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Segments Requested</span>
          <span className="stat-value">{measurement.num_segments_requested}</span>
        </div>
      </div>

      <div className="chart-container">
        <h3 className="chart-title">Segment Analysis</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart
            data={chartData}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis 
              dataKey="segment" 
              label={{ value: 'Segment', position: 'insideBottom', offset: -5 }}
              stroke="#4a5568"
            />
            <YAxis 
              label={{ value: 'X Coordinate', angle: -90, position: 'insideLeft' }}
              stroke="#4a5568"
            />
            <Tooltip 
              contentStyle={{
                background: 'rgba(255, 255, 255, 0.95)',
                border: 'none',
                borderRadius: '8px',
                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
              }}
            />
            <Legend 
              wrapperStyle={{
                paddingTop: '10px'
              }}
            />
            <Line 
              type="monotone" 
              dataKey="min_x" 
              stroke="#667eea" 
              strokeWidth={2}
              name="Min X"
              dot={{ fill: '#667eea', r: 4 }}
              activeDot={{ r: 6 }}
            />
            <Line 
              type="monotone" 
              dataKey="max_x" 
              stroke="#764ba2" 
              strokeWidth={2}
              name="Max X"
              dot={{ fill: '#764ba2', r: 4 }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="segments-table">
        <h4>Detailed Measurements</h4>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Segment</th>
                <th>Start Frame</th>
                <th>End Frame</th>
                <th>Min X</th>
                <th>Max X</th>
              </tr>
            </thead>
            <tbody>
              {measurement.segments.map(segment => (
                <tr key={segment.segment}>
                  <td>{segment.segment}</td>
                  <td>{segment.start_frame}</td>
                  <td>{segment.end_frame}</td>
                  <td>{segment.min_x ?? 'N/A'}</td>
                  <td>{segment.max_x ?? 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default MeasurementCard

