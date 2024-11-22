# %% Carga de bibliotecas
import os
from pathlib import Path

# %% Set WD and Local Libraries
os.chdir(Path(os.path.abspath("")).parent)
print(os.getcwd())

# %% Carga de funciones locales
from src.utils import save_frame_with_roi

# %% Ruta al video
video_path = "files/data/input/IMG_horizontal.mov"
output_path = "files/medias/frame_with_roi.jpg"
roi_ratios = (0.38, 0, 0.675, 1)
save_frame_with_roi(video_path, output_path, roi_ratios)
# %%
(1460, 0, 2590, 2160)