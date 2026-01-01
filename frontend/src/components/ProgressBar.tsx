import { motion } from 'framer-motion';
import { Loader2 } from 'lucide-react';

interface ProgressBarProps {
  progress: number;
  status: 'uploading' | 'processing';
}

export const ProgressBar = ({ progress, status }: ProgressBarProps) => {
  return (
    <motion.div
      className="progress-container"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
    >
      <div className="progress-header">
        <Loader2 className="spinning" size={24} />
        <span>
          {status === 'uploading' ? 'Uploading video...' : 'Analyzing video...'}
        </span>
      </div>

      <div className="progress-bar">
        <motion.div
          className="progress-fill"
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.3 }}
        />
      </div>

      <span className="progress-text">
        {status === 'uploading' ? `${progress}%` : 'This may take a few minutes...'}
      </span>
    </motion.div>
  );
};
