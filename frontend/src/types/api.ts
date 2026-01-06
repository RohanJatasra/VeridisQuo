export interface AnalysisOptions {
  frames_per_second: number;
  aggregation_method: 'majority' | 'average' | 'weighted_average' | 'max_confidence' | 'threshold';
  enable_gradcam: boolean;
  gradcam_alpha: number;
  batch_size: number;
  include_frame_results: boolean;
}

export interface FrameResult {
  prediction: 'FAKE' | 'REAL';
  confidence: number;
  probabilities: {
    FAKE: number;
    REAL: number;
  };
}

export interface AggregationMetadata {
  method: string;
  frame_count: number;
  fake_ratio?: number;
  real_ratio?: number;
  average_fake_confidence?: number;
  average_real_confidence?: number;
}

export interface AnalysisResponse {
  success: boolean;
  prediction: 'FAKE' | 'REAL';
  confidence: number;
  authenticity_score: number;
  num_frames_analyzed: number;
  aggregation: AggregationMetadata;
  frame_results?: FrameResult[];
  gradcam_video_url?: string;
  processing_time_seconds: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string;
  version: string;
}

export interface ErrorResponse {
  success: false;
  error: string;
  detail?: string;
}
