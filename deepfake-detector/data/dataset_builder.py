

from preprocessing.FramesExtractor import FramesExtractor
from preprocessing.FaceDetector import FaceDetector
from preprocessing.FaceExtractor import FaceExtractor
from data.dataset import CustomDataset, CSVFile
from config import (
    FACES_DIR, FRAME_DIR, DATA_FOLDER, CSV_FOLDER,
    TEST_FACES_DIR, TEST_FRAME_DIR, TEST_DATA_FOLDER, TEST_CSV_FOLDER
)
import os
import csv
import logging
from tqdm import tqdm
import cv2


class DatasetBuilder:
    def __init__(self, mode='train'):
        """
        Initialize DatasetBuilder with specified mode.

        Args:
            mode: 'train' for training dataset or 'test' for test dataset
        """
        if mode not in ['train', 'test']:
            raise ValueError(f"Mode must be 'train' or 'test', got '{mode}'")

        self.mode = mode

        # Set folders based on mode
        if mode == 'train':
            self.csv_folder = CSV_FOLDER
            self.data_folder = DATA_FOLDER
            self.frame_dir = FRAME_DIR
            self.faces_dir = FACES_DIR
        else:  # mode == 'test'
            self.csv_folder = TEST_CSV_FOLDER
            self.data_folder = TEST_DATA_FOLDER
            self.frame_dir = TEST_FRAME_DIR
            self.faces_dir = TEST_FACES_DIR

        self.dataset = CustomDataset(csv_folder=self.csv_folder)
        self.videos_path : dict[str, list[CSVFile]] = self.dataset.load_all_csvs()
        self.logger = logging.getLogger(f"DatasetBuilder[{mode}]")
        logging.basicConfig(level=logging.INFO)

    def extract_frame(self):
        """
        Extract frames from all videos and create a tracking CSV file.
        """
        # Create frames directory
        os.makedirs(self.frame_dir, exist_ok=True)
        

        # Prepare CSV data
        csv_data = []

        # Collect all videos to process
        all_videos = []
        for csv_name, video_list in self.videos_path.items():
            for video in video_list:
                all_videos.append((csv_name, video))

        self.logger.info(f"Starting frame extraction for {len(all_videos)} videos...")

        # Process videos with progress bar
        with tqdm(total=len(all_videos), desc="Extracting frames", unit="video") as pbar:
            for csv_name, video in all_videos:
                video_path = video.File_Path
                video_name = os.path.splitext(os.path.basename(video_path))[0]

                complete_video_path = os.path.join(self.data_folder, video_path)

                # Update progress bar description
                pbar.set_description(f"Processing {video_name}")

                # Skip if video file doesn't exist
                if not os.path.exists(complete_video_path):
                    self.logger.warning(f"Video not found: {complete_video_path}")
                    pbar.update(1)
                    continue

                try:
                    # Initialize frame extractor
                    video_extractor = FramesExtractor(complete_video_path, nb_fps=1)

                    # Create output directory for this video's frames
                    output_dir = os.path.join(self.frame_dir, video_name)
                    os.makedirs(output_dir, exist_ok=True)

                    # Extract and save frames
                    frame_count = 0
                    for frame, metadata in video_extractor.extract_frames():
                        # Generate frame filename
                        frame_filename = f"frame_{metadata['frame_number']:06d}.jpg"
                        frame_path = os.path.join(output_dir, frame_filename)

                        # Save frame
                        video_extractor.save_frame(frame, frame_path)

                        # Add to CSV data
                        csv_data.append({
                            "Frame_Path": frame_path,
                            "Video_Path": video_path,
                            "Label": video.Label,
                            "Width": video_extractor.width,
                            "Height": video_extractor.height,
                            "Frame_Number": metadata['frame_number'],
                            "Timestamp": metadata['timestamp'],
                            "Dataset": csv_name.replace('.csv', '')
                        })

                        frame_count += 1

                    pbar.set_postfix({"frames": frame_count, "dataset": csv_name.replace('.csv', '')})

                except Exception as e:
                    self.logger.error(f"✗ Error processing {video_path}: {str(e)}")
                finally:
                    pbar.update(1)

        # Save CSV
        self.logger.info(f"Saving metadata to {self.frame_dir}...")

        with open(os.path.join(self.frame_dir, "frames_metadata.csv"), 'w', newline='') as csvfile:
            if csv_data:
                fieldnames = ["Frame_Path", "Video_Path", "Label", "Width", "Height", "Frame_Number", "Timestamp", "Dataset"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)

        self.logger.info(f"✓ Frame extraction complete! {len(csv_data)} frames extracted from {len(all_videos)} videos")
        self.logger.info(f"✓ Metadata saved")

        return csv_data
    
    def extract_face(self):
        """
        Extract faces from frames using the frames_metadata.csv file.
        Creates a faces_metadata.csv file tracking all extracted faces.
        """
        # Create faces directory
        os.makedirs(self.faces_dir, exist_ok=True)

        # Path to frames metadata CSV
        frames_metadata_path = os.path.join(self.frame_dir, "frames_metadata.csv")

        # Check if frames metadata exists
        if not os.path.exists(frames_metadata_path):
            self.logger.error(f"Frames metadata not found at {frames_metadata_path}")
            self.logger.error("Please run extract_frame() first")
            return []

        # Initialize face detector and extractor
        face_detector = FaceDetector(
            min_face_size=40,
            confidence_threshold=0.7,
            only_keep_top=True
        )
        face_extractor = FaceExtractor(
            target_size=(224, 224),
            normalization="zero_one",
            padding=0.5
        )

        # Read frames metadata
        frames_data = []
        with open(frames_metadata_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            frames_data = list(reader)

        self.logger.info(f"Processing {len(frames_data)} frames for face extraction...")

        # Prepare CSV data for faces
        faces_csv_data = []
        total_faces_extracted = 0
        frames_with_no_faces = 0

        # Process frames with progress bar
        with tqdm(total=len(frames_data), desc="Extracting faces", unit="frame") as pbar:
            for frame_data in frames_data:
                frame_path = frame_data['Frame_Path']
                frame_number = int(frame_data['Frame_Number'])
                video_path = frame_data['Video_Path']
                label = frame_data['Label']
                dataset = frame_data['Dataset']

                pbar.set_description(f"Processing frame {frame_number}")

                # Check if frame exists
                if not os.path.exists(frame_path):
                    self.logger.warning(f"Frame not found: {frame_path}")
                    pbar.update(1)
                    continue

                try:
                    # Load frame
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        self.logger.warning(f"Could not load frame: {frame_path}")
                        pbar.update(1)
                        continue

                    # Detect faces
                    detections = face_detector.detect_faces(frame, return_landmarks=False)

                    if not detections:
                        frames_with_no_faces += 1
                        pbar.update(1)
                        continue

                    # Extract each detected face (usually just one due to only_keep_top=True)
                    for idx, detection in enumerate(detections):
                        bbox = detection['bbox']
                        confidence = detection['confidence']

                        # Extract and preprocess face
                        preprocessed_face, metadata = face_extractor.extract_and_preprocess(
                            frame,
                            bbox,
                            normalize=True
                        )

                        # Create output directory for faces from this video
                        video_name = os.path.splitext(os.path.basename(video_path))[0]
                        face_output_dir = os.path.join(self.faces_dir, video_name)
                        os.makedirs(face_output_dir, exist_ok=True)

                        # Generate face filename
                        face_filename = f"face_frame_{frame_number:06d}_{idx:02d}.jpg"
                        face_path = os.path.join(face_output_dir, face_filename)

                        # Save face
                        face_extractor.save(preprocessed_face, face_path, denormalize=True)

                        # Add to CSV data
                        faces_csv_data.append({
                            "Face_Path": face_path,
                            "Frame_Path": frame_path,
                            "Video_Path": video_path,
                            "Label": label,
                            "Frame_Number": frame_number,
                            "Dataset": dataset,
                            "Confidence": confidence,
                            "BBox_X": bbox[0],
                            "BBox_Y": bbox[1],
                            "BBox_Width": bbox[2],
                            "BBox_Height": bbox[3],
                            "Face_Width": face_extractor.target_size[0],
                            "Face_Height": face_extractor.target_size[1]
                        })

                        total_faces_extracted += 1

                    pbar.set_postfix({
                        "faces": total_faces_extracted,
                        "no_faces": frames_with_no_faces
                    })

                except Exception as e:
                    self.logger.error(f"Error processing frame {frame_path}: {str(e)}")
                finally:
                    pbar.update(1)

        # Save faces metadata CSV
        self.logger.info(f"Saving faces metadata to {self.faces_dir}...")

        faces_metadata_path = os.path.join(self.faces_dir, "faces_metadata.csv")
        with open(faces_metadata_path, 'w', newline='') as csvfile:
            if faces_csv_data:
                fieldnames = [
                    "Face_Path", "Frame_Path", "Video_Path", "Label",
                    "Frame_Number", "Dataset", "Confidence",
                    "BBox_X", "BBox_Y", "BBox_Width", "BBox_Height",
                    "Face_Width", "Face_Height"
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(faces_csv_data)

        self.logger.info(f"✓ Face extraction complete!")
        self.logger.info(f"✓ {total_faces_extracted} faces extracted from {len(frames_data)} frames")
        self.logger.info(f"✓ {frames_with_no_faces} frames had no detectable faces")
        self.logger.info(f"✓ Metadata saved to {faces_metadata_path}")

        return faces_csv_data