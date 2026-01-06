import { motion } from 'framer-motion';
import {
  CheckCircle,
  XCircle,
  Clock,
  Film,
  Download,
  RotateCcw,
  TrendingUp,
} from 'lucide-react';
import type { AnalysisResponse } from '../types/api';
import { getGradcamVideoUrl } from '../services/api';

interface ResultsProps {
  result: AnalysisResponse;
  onReset: () => void;
}

export const Results = ({ result, onReset }: ResultsProps) => {
  const isReal = result.prediction === 'REAL';
  const scoreColor = result.authenticity_score >= 70 ? '#10b981' : result.authenticity_score >= 40 ? '#f59e0b' : '#ef4444';

  // Calculate stroke-dashoffset for circular progress
  const circumference = 2 * Math.PI * 45;
  const strokeDashoffset = circumference - (result.authenticity_score / 100) * circumference;

  return (
    <motion.div
      className="results-container"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
    >
      {/* Main Score */}
      <div className="score-section">
        <div className="score-ring">
          <svg viewBox="0 0 100 100">
            <circle className="score-bg" cx="50" cy="50" r="45" />
            <motion.circle
              className="score-progress"
              cx="50"
              cy="50"
              r="45"
              style={{ stroke: scoreColor }}
              initial={{ strokeDashoffset: circumference }}
              animate={{ strokeDashoffset }}
              transition={{ duration: 1.5, ease: 'easeOut' }}
            />
          </svg>
          <div className="score-value">
            <motion.span
              className="score-number"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
            >
              {Math.round(result.authenticity_score)}
            </motion.span>
            <span className="score-label">Authenticity</span>
          </div>
        </div>

        <motion.div
          className={`verdict ${isReal ? 'real' : 'fake'}`}
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          {isReal ? (
            <CheckCircle size={32} className="verdict-icon" />
          ) : (
            <XCircle size={32} className="verdict-icon" />
          )}
          <span className="verdict-text">
            {isReal ? 'Likely Authentic' : 'Likely Manipulated'}
          </span>
          <span className="confidence">
            {(result.confidence * 100).toFixed(1)}% confidence
          </span>
        </motion.div>
      </div>

      {/* Stats Grid */}
      <div className="stats-grid">
        <div className="stat-card">
          <Film size={24} />
          <div className="stat-content">
            <span className="stat-value">{result.num_frames_analyzed}</span>
            <span className="stat-label">Frames Analyzed</span>
          </div>
        </div>

        <div className="stat-card">
          <Clock size={24} />
          <div className="stat-content">
            <span className="stat-value">{result.processing_time_seconds.toFixed(1)}s</span>
            <span className="stat-label">Processing Time</span>
          </div>
        </div>

        <div className="stat-card">
          <TrendingUp size={24} />
          <div className="stat-content">
            <span className="stat-value">{result.aggregation.method}</span>
            <span className="stat-label">Aggregation</span>
          </div>
        </div>
      </div>

      {/* Probability Bars */}
      <div className="probabilities">
        <h4>Detection Confidence</h4>
        <div className="prob-bars">
          <div className="prob-item">
            <span className="prob-label">REAL</span>
            <div className="prob-bar">
              <motion.div
                className="prob-fill real"
                initial={{ width: 0 }}
                animate={{
                  width: `${((1 - result.confidence) * 100).toFixed(1)}%`,
                }}
                transition={{ duration: 0.8, delay: 0.5 }}
              />
            </div>
            <span className="prob-value">
              {result.prediction === 'REAL'
                ? (result.confidence * 100).toFixed(1)
                : ((1 - result.confidence) * 100).toFixed(1)}%
            </span>
          </div>
          <div className="prob-item">
            <span className="prob-label">FAKE</span>
            <div className="prob-bar">
              <motion.div
                className="prob-fill fake"
                initial={{ width: 0 }}
                animate={{
                  width: `${(result.prediction === 'FAKE' ? result.confidence * 100 : (1 - result.confidence) * 100).toFixed(1)}%`,
                }}
                transition={{ duration: 0.8, delay: 0.6 }}
              />
            </div>
            <span className="prob-value">
              {result.prediction === 'FAKE'
                ? (result.confidence * 100).toFixed(1)
                : ((1 - result.confidence) * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      {/* GradCAM Video */}
      {result.gradcam_video_url && (
        <motion.div
          className="gradcam-section"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
        >
          <h4>GradCAM Visualization</h4>
          <p className="gradcam-hint">
            Red areas indicate regions the model found suspicious
          </p>
          <video
            src={getGradcamVideoUrl(result.gradcam_video_url)}
            controls
            className="gradcam-video"
          />
          <a
            href={getGradcamVideoUrl(result.gradcam_video_url)}
            download
            className="download-btn"
          >
            <Download size={18} />
            Download GradCAM Video
          </a>
        </motion.div>
      )}

      {/* Frame Results Chart */}
      {result.frame_results && result.frame_results.length > 0 && (
        <motion.div
          className="frames-section"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
        >
          <h4>Per-Frame Analysis</h4>
          <div className="frames-chart">
            {result.frame_results.map((frame, index) => (
              <div
                key={index}
                className={`frame-bar ${frame.prediction.toLowerCase()}`}
                style={{ height: `${frame.confidence * 100}%` }}
                title={`Frame ${index + 1}: ${frame.prediction} (${(frame.confidence * 100).toFixed(1)}%)`}
              />
            ))}
          </div>
          <div className="frames-legend">
            <span><span className="legend-dot real" /> Real</span>
            <span><span className="legend-dot fake" /> Fake</span>
          </div>
        </motion.div>
      )}

      {/* Actions */}
      <div className="actions">
        <button className="btn-secondary" onClick={onReset}>
          <RotateCcw size={18} />
          Analyze Another Video
        </button>
      </div>
    </motion.div>
  );
};
