import os
import cv2
import numpy as np
from ultralytics import YOLO
from google.cloud import vision
from src.sort.sort import Sort
from tqdm import tqdm

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'key.json'
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = 'true'

class CarLicensePlateDetector:
    """
    A class to detect and recognize license plates on cars using the YOLO model and OCR.

    Attributes:
        model (YOLO): An instance of the YOLO object detection model.
    """

    def __init__(self, vehicles_path: str, placas_path: str):
        """
        Initializes the CarLicensePlateDetector with the given weights.

        Args:
            weights_path (str): The path to the weights file for the YOLO model.
        """
        self.model_vehicles = YOLO(vehicles_path, task='detect')
        self.model_placas = YOLO(placas_path, task='detect')
        
    def process_video(self, video_path: str, output_path: str, roi_ratios: tuple = None) -> None:
        """
        Processes a video file to detect and recognize license plates in each frame.

        Args:
            video_path (str): The path to the video file.
            output_path (str): The path where the output video will be saved.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Error opening video file")
        
        # Get the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        results = {}
        mot_tracker = Sort()
        vehicles = [2, 3, 5, 7]
        frame_nmr = -1
        ret = True
        
        # Initialize tqdm progress bar
        with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
            while ret:
                frame_nmr += 1
                ret, frame = cap.read()
                if ret:
                    results[frame_nmr] = {}
                    # detect vehicles
                    detections = self.model_vehicles(frame, verbose=False)[0]
                    detections_ = []
                    for detection in detections.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = detection
                        if int(class_id) in vehicles:
                            # Para limitar el análisis a un sector
                            if roi_ratios:
                                roi_x1, roi_y1, roi_x2, roi_y2 = roi_ratios
                                bbox_center_x = (x1 + x2) / 2
                                bbox_center_y = (y1 + y2) / 2
                                if roi_x1 <= bbox_center_x <= roi_x2 and roi_y1 <= bbox_center_y <= roi_y2:
                                    detections_.append([x1, y1, x2, y2, score])
                            else:
                                detections_.append([x1, y1, x2, y2, score])
                    # track vehicles
                    track_ids = mot_tracker.update(np.asarray(detections_))
                    
                    # detect license plates
                    license_plates = self.model_placas(frame, verbose=False)[0]

                    for license_plate in license_plates.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = license_plate

                        # assign license plate to car
                        xcar1, ycar1, xcar2, ycar2, car_id = self.get_car(license_plate, track_ids)

                        if car_id != -1:

                            # crop license plate
                            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                            # process license plate
                            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                            # read license plate number
                            license_plate_text, license_plate_text_score = self.extract_license_plate_text_with_vision(license_plate_crop_thresh)

                            if license_plate_text is not None:
                                results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                            'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                                'text': license_plate_text,
                                                                                'bbox_score': score,
                                                                                'text_score': license_plate_text_score}}
                    
                    # Update tqdm progress bar
                    pbar.update(1)
        
        # Write results
        self.write_csv(results, output_path)
        
    def get_car(self, license_plate, vehicle_track_ids):
        """
        Retrieve the vehicle coordinates and ID based on the license plate coordinates.

        Args:
            license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
            vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

        Returns:
            tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
        """
        x1, y1, x2, y2, score, class_id = license_plate

        foundIt = False
        for j in range(len(vehicle_track_ids)):
            xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

            if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
                car_indx = j
                foundIt = True
                break

        if foundIt:
            return vehicle_track_ids[car_indx]

        return -1, -1, -1, -1, -1

    def write_csv(self, results, output_path):
        """
        Write the results to a CSV file.

        Args:
            results (dict): Dictionary containing the results.
            output_path (str): Path to the output CSV file.
        """
        with open(output_path, 'w') as f:
            f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                    'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                    'license_number_score'))

            for frame_nmr in results.keys():
                for car_id in results[frame_nmr].keys():
                    if 'car' in results[frame_nmr][car_id].keys() and \
                    'license_plate' in results[frame_nmr][car_id].keys() and \
                    'text' in results[frame_nmr][car_id]['license_plate'].keys():
                        f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                                car_id,
                                                                '[{} {} {} {}]'.format(
                                                                    results[frame_nmr][car_id]['car']['bbox'][0],
                                                                    results[frame_nmr][car_id]['car']['bbox'][1],
                                                                    results[frame_nmr][car_id]['car']['bbox'][2],
                                                                    results[frame_nmr][car_id]['car']['bbox'][3]),
                                                                '[{} {} {} {}]'.format(
                                                                    results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                    results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                    results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                    results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                                results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                                results[frame_nmr][car_id]['license_plate']['text'],
                                                                results[frame_nmr][car_id]['license_plate']['text_score'])
                                )
            f.close()
            
    @staticmethod
    def extract_license_plate_text_with_vision(roi: np.ndarray) -> tuple[str, float]:
        """
        Extracts the text and confidence score from a region of interest (ROI) using Google Cloud Vision API.

        Args:
            roi (np.ndarray): The region of interest containing the license plate.

        Returns:
            tuple[str, float]: A tuple containing the recognized text and its confidence score.
        """
        # Convert the ROI to bytes for the Vision API
        _, encoded_image = cv2.imencode('.jpg', roi)
        roi_bytes = encoded_image.tobytes()

        # Initialize the Google Cloud Vision client
        client = vision.ImageAnnotatorClient()

        # Prepare the image for the Vision API
        image = vision.Image(content=roi_bytes)

        # Perform document text detection instead of simple text detection
        response = client.document_text_detection(image=image)

        # Check for errors
        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(response.error.message)
            )

        # Extract text and confidence scores
        if response.full_text_annotation:
            # Get all pages (usually just one for an image)
            text_blocks = []
            confidence_scores = []
            
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        paragraph_text = ""
                        paragraph_confidence = []
                        
                        for word in paragraph.words:
                            word_text = ''.join([
                                symbol.text for symbol in word.symbols
                            ])
                            word_confidence = sum(
                                symbol.confidence for symbol in word.symbols
                            ) / len(word.symbols)
                            
                            paragraph_text += word_text + " "
                            paragraph_confidence.append(word_confidence)
                        
                        text_blocks.append(paragraph_text.strip())
                        # Average confidence for the paragraph
                        confidence_scores.append(
                            sum(paragraph_confidence) / len(paragraph_confidence)
                            if paragraph_confidence else 0.0
                        )
            
            if text_blocks:
                # Join all text blocks and calculate average confidence
                final_text = " ".join(text_blocks)
                final_confidence = sum(confidence_scores) / len(confidence_scores)
                return final_text, final_confidence
                
        return "", 0.0
