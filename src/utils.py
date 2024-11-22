import cv2
import os

def convert_to_linux_path(path):
    if os.name == 'posix' and '\\' in path:
        # Convierte barras invertidas de Windows a barras normales de Linux
        path = path.replace('\\', '/')
    return path

def save_frame_with_roi(video_path, output_path, roi_ratios):
    """
    Processes the first frame of a video, draws a rectangle for the ROI, 
    saves the frame, and prints the ROI coordinates.
    
    Args:
        video_path (str): Path to the video file.
        output_path (str): Path to save the frame with the ROI.
        roi_ratios (tuple): Ratios for the ROI (x1_ratio, y1_ratio, x2_ratio, y2_ratio).
    """
    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: No se puede abrir el video.")
        return
    
    # Leer el primer frame
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el primer frame.")
        cap.release()
        return

    # Obtener dimensiones del frame
    frame_height, frame_width = frame.shape[:2]

    # Calcular coordenadas del ROI
    x1_ratio, y1_ratio, x2_ratio, y2_ratio = roi_ratios
    roi_x1, roi_y1 = int(frame_width * x1_ratio), int(frame_height * y1_ratio)
    roi_x2, roi_y2 = int(frame_width * x2_ratio), int(frame_height * y2_ratio)

    # Dibujar el ROI en el frame
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)

    # Guardar el frame con el ROI
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Crear directorio si no existe
    cv2.imwrite(output_path, frame)

    # Imprimir las coordenadas del ROI
    print(f"Dimensiones del frame: {frame_width}x{frame_height}")
    print(f"Coordenadas del ROI: ({roi_x1}, {roi_y1}) a ({roi_x2}, {roi_y2})")
    print(f"Frame guardado en: {output_path}")

    # Liberar recursos
    cap.release()
