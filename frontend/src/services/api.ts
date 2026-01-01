import axios from 'axios';
import type { AxiosProgressEvent } from 'axios';
import type { AnalysisOptions, AnalysisResponse, HealthResponse } from '../types/api';

const API_HOST = import.meta.env.VITE_API_HOST || 'http://localhost:8000';
const API_BASE_URL = `${API_HOST}/api/v1`;

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 600000, // 10 minutes for video processing
});

export const checkHealth = async (): Promise<HealthResponse> => {
  const response = await api.get<HealthResponse>('/health');
  return response.data;
};

export const analyzeVideo = async (
  file: File,
  options: Partial<AnalysisOptions>,
  onProgress?: (progress: number) => void
): Promise<AnalysisResponse> => {
  const formData = new FormData();
  formData.append('video', file);

  // Append options
  if (options.frames_per_second) {
    formData.append('frames_per_second', options.frames_per_second.toString());
  }
  if (options.aggregation_method) {
    formData.append('aggregation_method', options.aggregation_method);
  }
  if (options.enable_gradcam !== undefined) {
    formData.append('enable_gradcam', options.enable_gradcam.toString());
  }
  if (options.gradcam_alpha) {
    formData.append('gradcam_alpha', options.gradcam_alpha.toString());
  }
  if (options.batch_size) {
    formData.append('batch_size', options.batch_size.toString());
  }
  if (options.include_frame_results !== undefined) {
    formData.append('include_frame_results', options.include_frame_results.toString());
  }

  const response = await api.post<AnalysisResponse>('/analyze', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent: AxiosProgressEvent) => {
      if (progressEvent.total && onProgress) {
        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(progress);
      }
    },
  });

  return response.data;
};

export const getGradcamVideoUrl = (path: string): string => {
  // path already contains /api/v1/outputs/..., so use API_HOST directly
  return `${API_HOST}${path}`;
};

export default api;
