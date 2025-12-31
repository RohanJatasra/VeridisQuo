import { useCallback, useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Film, X, FileVideo, AlertTriangle } from 'lucide-react';

interface VideoUploadProps {
  file: File | null;
  onFileSelect: (file: File | null) => void;
  disabled?: boolean;
}

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export const VideoUpload = ({ file, onFileSelect, disabled }: VideoUploadProps) => {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState(false);

  // Create and cleanup object URL for video preview
  useEffect(() => {
    if (file) {
      const url = URL.createObjectURL(file);
      setVideoUrl(url);
      setPreviewError(false);
      return () => URL.revokeObjectURL(url);
    } else {
      setVideoUrl(null);
      setPreviewError(false);
    }
  }, [file]);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onFileSelect(acceptedFiles[0]);
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
    },
    maxFiles: 1,
    disabled,
  });

  const removeFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    onFileSelect(null);
  };

  const rootProps = getRootProps();

  return (
    <div className="upload-container">
      <motion.div
        onClick={rootProps.onClick}
        onKeyDown={rootProps.onKeyDown}
        onFocus={rootProps.onFocus}
        onBlur={rootProps.onBlur}
        tabIndex={rootProps.tabIndex}
        role={rootProps.role}
        className={`dropzone ${isDragActive ? 'active' : ''} ${file ? 'has-file' : ''} ${disabled ? 'disabled' : ''}`}
        whileHover={!disabled ? { scale: 1.01 } : {}}
        whileTap={!disabled ? { scale: 0.99 } : {}}
      >
        <input {...getInputProps()} />

        <AnimatePresence mode="wait">
          {!file ? (
            <motion.div
              key="upload-prompt"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="upload-prompt"
            >
              {isDragActive ? (
                <>
                  <Film size={64} className="upload-icon active" />
                  <h3>Drop your video here</h3>
                </>
              ) : (
                <>
                  <Upload size={64} className="upload-icon" />
                  <h3>Drag & drop your video</h3>
                  <p>or click to browse</p>
                  <span className="supported-formats">
                    MP4, AVI, MOV, MKV, WEBM
                  </span>
                </>
              )}
            </motion.div>
          ) : (
            <motion.div
              key="file-preview"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="file-preview"
            >
              <div className="file-info">
                <FileVideo size={48} className="file-icon" />
                <div className="file-details">
                  <span className="file-name">{file.name}</span>
                  <span className="file-size">{formatFileSize(file.size)}</span>
                </div>
                {!disabled && (
                  <button className="remove-btn" onClick={removeFile}>
                    <X size={20} />
                  </button>
                )}
              </div>
              {previewError ? (
                <div className="preview-error">
                  <AlertTriangle size={32} />
                  <span>Preview unavailable (video will still be analyzed)</span>
                </div>
              ) : videoUrl && (
                <video
                  src={videoUrl}
                  className="video-thumbnail"
                  muted
                  playsInline
                  onError={() => setPreviewError(true)}
                />
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
};
