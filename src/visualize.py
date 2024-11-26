import ast

import cv2
import pandas as pd
import numpy as np

from tqdm import tqdm

def process_video_with_license_plates(video_path: str, results: pd.DataFrame, video_output_path: str) -> None:
    """
    Process a video file to draw bounding boxes around cars and their license plates,
    and display license plate information.

    Args:
        video_path (str): Path to the input video file
        results (pd.DataFrame): DataFrame containing detection results with columns:
            - car_id
            - frame_nmr
            - car_bbox
            - license_plate_bbox
            - license_number
            - license_number_score
        output_path (str): Path where the processed video will be saved
    """
    def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
        x1, y1 = top_left
        x2, y2 = bottom_right

        cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
        cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

        cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
        cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

        cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
        cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

        cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
        cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

        return img

    # load video and interpolate data
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    license_plate = {}
    for car_id in np.unique(results['car_id']):
        max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
        license_plate[car_id] = {'license_crop': None,
                                'license_plate_number': results[(results['car_id'] == car_id) &
                                                                (results['license_number_score'] == max_)]['license_number'].iloc[0]}
        cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                                (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
        ret, frame = cap.read()

        x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                                (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

        license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        # license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
        # Cambia la proporción de la redimensión: Ancho mayor, altura menor
        license_crop = cv2.resize(license_crop, (int((x2 - x1) * 380 / (y2 - y1)), 200))

        license_plate[car_id]['license_crop'] = license_crop


    frame_nmr = -1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    with tqdm(total=total_frames, desc="Processing video frames") as pbar:
        # read frames
        ret = True
        while ret:
            ret, frame = cap.read()
            frame_nmr += 1
            cv2.rectangle(frame, (1460, 0), (2590, 2160), (0, 255, 0), 2)
            if ret:
                df_ = results[results['frame_nmr'] == frame_nmr]
                for row_indx in range(len(df_)):
                    # draw car
                    car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                    draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 10,
                                line_length_x=200, line_length_y=200)

                    # draw license plate
                    x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)

                    # crop license plate
                    license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

                    H, W, _ = license_crop.shape

                    try:
                        frame[int(car_y1) - H - 50:int(car_y1) - 50,
                            int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                        frame[int(car_y1) - H - 160:int(car_y1) - H - 50,
                            int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                        (text_width, text_height), _ = cv2.getTextSize(
                            license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2, # Reducir escala de fuente
                            5 # Reducir grosor del texto
                        )
                        
                        center_y = int((int(car_y1) - H - 160 + int(car_y1) - H - 50) / 2)
                        cv2.putText(frame,
                                    license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                                    (int((car_x2 + car_x1 - text_width) / 2), int(center_y + (text_height / 2))),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    2, # Reducir escala de fuente
                                    (0, 0, 0),
                                    5 # Reducir grosor del texto
                                )

                    except:
                        pass

                out.write(frame)
                frame = cv2.resize(frame, (1280, 720))
            pbar.update(1)

    out.release()
    cap.release()