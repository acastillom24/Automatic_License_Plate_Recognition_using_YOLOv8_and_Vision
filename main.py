from src.inference import CarLicensePlateDetector
from src.add_missing_data import interpolate_bounding_boxes
from src.visualize import process_video_with_license_plates
from src.utils import convert_to_linux_path
import yaml
import csv
import pandas as pd
import os

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

PATH_MODEL_VEHICLES = config["input"]["path_model_vehicles"]
PATH_MODEL_PLACAS = config["input"]["path_model_placas"]
PATH_VIDEO_INPUT = config["input"]["path_video"]
PATH_VIDEO_OUTPUT_CSV = config["output"]["path_video_csv"]
PATH_VIDEO_INTERPOLATED_OUTPUT_CSV = config["output"]["path_video_interpolated_csv"]
PATH_VIDEO_OUTPUT = config["output"]["path_video"]
        
def main_menu():
    while True:
        print("\nMenú Principal:")
        print("1. Detectar las placas en los vehículos")
        print("2. Interpolar los datos pedidos")
        print("3. Procesamiento del video")
        print("4. Salir")
        choice = input("Seleccione una opción: ")

        if choice == '1':
            path_video = input("Ingrese la ruta del video a analizar: ")
            path_video = convert_to_linux_path(path_video)
            file_name = os.path.splitext(os.path.basename(path_video))[0]
            path_csv = f'files/data/output/{file_name}.csv'
            
            if not os.path.isfile(path_video):
                print("El video no existe o no es accesible")
                return
            
            print("Start processing ....")
            detector = CarLicensePlateDetector(PATH_MODEL_VEHICLES, PATH_MODEL_PLACAS)
            roi_ratios = (1460, 0, 2590, 2160)
            detector.process_video(path_video, path_csv, roi_ratios)
            
        elif choice == '2':
            path_csv = input("Ingrese la ruta del archivo csv a interpolar: ")
            path_csv = convert_to_linux_path(path_csv)
            file_name = os.path.splitext(os.path.basename(path_csv))[0]
            path_csv_interpolate = f'files/data/output/{file_name}_interpolate.csv'

            if not os.path.isfile(path_csv):
                print("El archivo csv no existe o no es accesible")
                return
            
            print("Start processing ....")
            
            with open(path_csv, 'r') as file:
                reader = csv.DictReader(file)
                data = list(reader)
                
            interpolated_data = interpolate_bounding_boxes(data)

            # Write updated data to a new CSV file
            header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
            with open(path_csv_interpolate, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=header)
                writer.writeheader()
                writer.writerows(interpolated_data)
            
        elif choice == '3':
            path_video = input("Ingrese la ruta del video a analizar: ")
            path_video = convert_to_linux_path(path_video)

            if not os.path.isfile(path_video):
                print("El video no existe o no es accesible")
                return
            
            path_csv_interpolate = input("Ingrese la ruta del csv interpolado: ")
            path_csv_interpolate = convert_to_linux_path(path_csv_interpolate)

            if not os.path.isfile(path_csv_interpolate):
                print("El csv interpolado no existe o no es accesible")
                return
            
            print("Start processing ....")
            dtypes = {
                "license_number_score": float
            }
            
            results = pd.read_csv(path_csv_interpolate, dtype=dtypes)
            file_name = os.path.splitext(os.path.basename(path_video))[0]
            path_video_output = f'files/data/output/{file_name}.mp4'
            process_video_with_license_plates(
                video_path=path_video,
                results=results,
                video_output_path=path_video_output
            )
        
        elif choice == '4':
            print("Saliendo del programa.")
            break

        else:
            print("Opción inválida. Intente de nuevo.")

if __name__ == '__main__':
    main_menu()
