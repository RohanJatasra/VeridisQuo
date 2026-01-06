import { useState, useEffect } from 'react';
import { AnimatePresence } from 'framer-motion';
import {
  Header,
  VideoUpload,
  AnalysisOptions,
  ProgressBar,
  Results,
} from './components';
import { checkHealth, analyzeVideo } from './services/api';
import type { AnalysisOptions as AnalysisOptionsType, AnalysisResponse, HealthResponse } from './types/api';
import { AlertCircle } from 'lucide-react';
import './App.css';

type AppState = 'idle' | 'uploading' | 'processing' | 'success' | 'error';

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [options, setOptions] = useState<Partial<AnalysisOptionsType>>({
    frames_per_second: 1,
    aggregation_method: 'weighted_average',
    enable_gradcam: true,
    include_frame_results: false,
  });
  const [state, setState] = useState<AppState>('idle');
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Check API health on mount
  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        const healthData = await checkHealth();
        setHealth(healthData);
      } catch (err) {
        console.error('API health check failed:', err);
        setHealth(null);
      }
    };

    checkApiHealth();
    const interval = setInterval(checkApiHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleAnalyze = async () => {
    if (!file) return;

    setState('uploading');
    setProgress(0);
    setError(null);

    try {
      const response = await analyzeVideo(file, options, (uploadProgress) => {
        setProgress(uploadProgress);
        if (uploadProgress === 100) {
          setState('processing');
        }
      });

      setResult(response);
      setState('success');
    } catch (err: unknown) {
      console.error('Analysis failed:', err);
      const errorMessage = err instanceof Error ? err.message : 'Analysis failed. Please try again.';
      setError(errorMessage);
      setState('error');
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError(null);
    setState('idle');
    setProgress(0);
  };

  const isProcessing = state === 'uploading' || state === 'processing';

  return (
    <div className="app">
      <Header health={health} />

      <main className="main">
        <AnimatePresence mode="wait">
          {state === 'success' && result ? (
            <Results key="results" result={result} onReset={handleReset} />
          ) : state === 'error' ? (
            <div key="error" className="error-container">
              <AlertCircle size={48} />
              <h3>Analysis Failed</h3>
              <p>{error}</p>
              <button className="btn-primary" onClick={handleReset}>
                Try Again
              </button>
            </div>
          ) : (
            <div key="upload" className="upload-flow">
              <VideoUpload
                file={file}
                onFileSelect={setFile}
                disabled={isProcessing}
              />

              {file && !isProcessing && (
                <AnalysisOptions
                  options={options}
                  onChange={setOptions}
                  disabled={isProcessing}
                />
              )}

              {isProcessing && (
                <ProgressBar
                  progress={progress}
                  status={state as 'uploading' | 'processing'}
                />
              )}

              {file && !isProcessing && (
                <button
                  className="btn-primary analyze-btn"
                  onClick={handleAnalyze}
                  disabled={!health?.model_loaded}
                >
                  {health?.model_loaded ? 'Analyze Video' : 'Model Loading...'}
                </button>
              )}
            </div>
          )}
        </AnimatePresence>
      </main>

      <footer className="footer">
        <p>VeridisQuo - Neural Network for Deepfake Detection</p>
        <div className="footer-links">
          <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer">
            API Docs
          </a>
          <span>|</span>
          <a href="https://github.com/VeridisQuo-orga" target="_blank" rel="noopener noreferrer">
            GitHub
          </a>
        </div>
      </footer>
    </div>
  );
}

export default App;
