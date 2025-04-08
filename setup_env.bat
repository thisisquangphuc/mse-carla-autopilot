@echo off
setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION

:: -- Directories
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_DIR=%%~fI"

:: -- Files
set "INI_FILE=%SCRIPT_DIR%settings.ini"
set "REQ_FILE=%SCRIPT_DIR%requirements.txt"
set "VENV_DIR=%PROJECT_DIR%\venv"

:: -- Variables
set "CARLA_PATH="
set "found_section=false"


:: -- Parse settings.ini
for /f "usebackq tokens=1,* delims==" %%A in ("%INI_FILE%") do (
    set "line=%%A"
    set "value=%%B"

    :: Trim leading spaces from the key
    for /f "tokens=* delims= " %%L in ("!line!") do set "line=%%L"

    :: Check which section we are in
    if /i "!line!"=="[CARLA-SERVER]" (
        set "found_section=true"
    ) else if /i "!line!"=="[CARLA-CLIENT]" (
        set "found_section=false"
    ) else (
        :: If we are in [CARLA-SERVER], look for CARLA_PATH
        if "!found_section!"=="true" (
            if /i "!line!"=="CARLA_PATH_WHL" (
                :: Trim leading spaces from the value
                for /f "tokens=* delims= " %%V in ("!value!") do set "CARLA_PATH=%%V"
            )
        )
    )
)

echo.
if "!CARLA_PATH!"=="" (
    echo ERROR: CARLA_PATH not found in settings.ini.
    exit /b 1
)

echo CARLA_PATH: !CARLA_PATH!
echo Creating virtual environment in: "%VENV_DIR%"

py -3.7 -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment. Make sure Python 3.7 is installed.
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate the virtual environment.
    exit /b 1
)

:: -- Install the CARLA egg file
echo Installing CARLA wheel from: !CARLA_PATH!
pip install !CARLA_PATH!
if errorlevel 1 (
    echo ERROR: Failed to install CARLA egg: !CARLA_PATH!
    exit /b 1
)

pip install -r "%REQ_FILE%"

echo.
echo Done. Changing directory to "%PROJECT_DIR%"
cd "%PROJECT_DIR%"
