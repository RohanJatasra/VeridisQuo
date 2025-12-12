

from preprocessing.FramesExtractor import FramesExtractor
from data.dataset import CustomDataset, CSVFile
from config import FACES_DIR, FRAME_DIR, DATA_FOLDER
import os
import csv
import logging
from tqdm import tqdm


class DatasetBuilder:
    def __init__(self):
        self.dataset = CustomDataset()
        self.videos_path : dict[str, list[CSVFile]] = self.dataset.load_all_csvs()
        self.logger = logging.getLogger("DatasetBuilder")
        logging.basicConfig(level=logging.INFO)

    def extract_frame(self):
        """
        Extract frames from all videos and create a tracking CSV file.

        Args:
            output_csv_path: Path to save the tracking CSV. If None, saves to data/frames/frames_metadata.csv
        """


        # Create frames directory
        os.makedirs(FRAME_DIR, exist_ok=True)
        

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

                complete_video_path = os.path.join(DATA_FOLDER, video_path)

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
                    output_dir = os.path.join(FRAME_DIR, video_name)
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
        self.logger.info(f"Saving metadata to {FRAME_DIR}...")

        with open(os.path.join(FRAME_DIR, "frames_metadata.csv"), 'w', newline='') as csvfile:
            if csv_data:
                fieldnames = ["Frame_Path", "Video_Path", "Label", "Width", "Height", "Frame_Number", "Timestamp", "Dataset"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)

        self.logger.info(f"✓ Frame extraction complete! {len(csv_data)} frames extracted from {len(all_videos)} videos")
        self.logger.info(f"✓ Metadata saved")

        return csv_data
    
    def extract_face(self):
        pass