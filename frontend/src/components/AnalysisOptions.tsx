import { motion } from 'framer-motion';
import { Settings, Zap, Eye, BarChart3 } from 'lucide-react';
import type { AnalysisOptions as AnalysisOptionsType } from '../types/api';

interface AnalysisOptionsProps {
  options: Partial<AnalysisOptionsType>;
  onChange: (options: Partial<AnalysisOptionsType>) => void;
  disabled?: boolean;
}

export const AnalysisOptions = ({ options, onChange, disabled }: AnalysisOptionsProps) => {
  return (
    <motion.div
      className="options-panel"
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      exit={{ opacity: 0, height: 0 }}
    >
      <div className="options-header">
        <Settings size={20} />
        <h3>Analysis Options</h3>
      </div>

      <div className="options-grid">
        {/* FPS Slider */}
        <div className="option-group">
          <label htmlFor="fps">
            <Zap size={16} />
            Frames per Second
          </label>
          <div className="slider-container">
            <input
              type="range"
              id="fps"
              min={1}
              max={10}
              value={options.frames_per_second || 1}
              onChange={(e) =>
                onChange({ ...options, frames_per_second: parseInt(e.target.value) })
              }
              disabled={disabled}
            />
            <span className="slider-value">{options.frames_per_second || 1} fps</span>
          </div>
        </div>

        {/* Aggregation Method */}
        <div className="option-group">
          <label htmlFor="aggregation">
            <BarChart3 size={16} />
            Aggregation Method
          </label>
          <select
            id="aggregation"
            value={options.aggregation_method || 'weighted_average'}
            onChange={(e) =>
              onChange({
                ...options,
                aggregation_method: e.target.value as AnalysisOptionsType['aggregation_method'],
              })
            }
            disabled={disabled}
          >
            <option value="weighted_average">Weighted Average (Recommended)</option>
            <option value="majority">Majority Vote</option>
            <option value="average">Simple Average</option>
            <option value="max_confidence">Max Confidence</option>
          </select>
        </div>

        {/* Checkboxes */}
        <div className="option-group checkbox">
          <label>
            <input
              type="checkbox"
              checked={options.enable_gradcam ?? true}
              onChange={(e) =>
                onChange({ ...options, enable_gradcam: e.target.checked })
              }
              disabled={disabled}
            />
            <Eye size={16} />
            Generate GradCAM visualization
          </label>
        </div>

        <div className="option-group checkbox">
          <label>
            <input
              type="checkbox"
              checked={options.include_frame_results ?? false}
              onChange={(e) =>
                onChange({ ...options, include_frame_results: e.target.checked })
              }
              disabled={disabled}
            />
            <BarChart3 size={16} />
            Include per-frame results
          </label>
        </div>
      </div>
    </motion.div>
  );
};
