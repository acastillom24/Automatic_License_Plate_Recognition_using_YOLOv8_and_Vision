# Automatic_License_Plate_Recognition_using_YOLOv8_and_Vision

## Configuración necesaria

- Cree un entorno virtual y activalo
```bash
python -m virtualenv venv
source venv/bin/activate
```

- Intale las bibliotecas necesarias
```bash
python -m pip install -r requirements.txt
```

- Instale el modulo [sort](https://github.com/abewley/sort?tab=readme-ov-file)

    - Clone el repositorio
    ```bash
    git clone https://github.com/abewley/sort.git
    ```
    - Luego:
    ```bash
    $ cd path/to/sort
    $ python sort.py
    ```

- Configure el archivo `config.yaml`, la opción `input`, modificar según sus rutas.

        path_model_vehicles
        path_model_placas

## Ejecutar el programa
```bash
python main.py
```
![menu](/menu.png)


## Evitar incluir bibliotecas innecesarias en el archivo requirements

- Instala `pipreqs` si no lo tienes
```bash
pip install pipreqs
```

- Navega hasta el directorio de tu proyecto y ejecuta
```bash
pipreqs . --force
```

- Para evitar el análisis en carpetas específicas del proyecto
```bash
pipreqs . --force --ignore sandbox,files,venv,src/sort,src/__pycache__
```

- Otra forma es especificando bibliotecas específicas
```bash
!pip list | grep -E 'tensorflow|keras' | awk '{print $1"=="$2}' > requirements.txt
```
