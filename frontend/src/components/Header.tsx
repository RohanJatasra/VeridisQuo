import { motion } from 'framer-motion';
import { Cpu } from 'lucide-react';
import type { HealthResponse } from '../types/api';

interface HeaderProps {
  health: HealthResponse | null;
}

export const Header = ({ health }: HeaderProps) => {
  return (
    <motion.header
      className="header"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="logo">
        <img src="/veridisquo_nobg.png" alt="VeridisQuo" className="logo-icon" />
        <div className="logo-text">
          <h1>VeridisQuo</h1>
          <span>AI-Powered Deepfake Detection</span>
        </div>
      </div>

      <div className={`status ${health?.model_loaded ? 'online' : 'offline'}`}>
        <Cpu size={16} />
        <span>
          {health?.model_loaded
            ? `Model ready (${health.device})`
            : 'Connecting...'}
        </span>
        <span className="status-dot" />
      </div>
    </motion.header>
  );
};
