@echo off
REM Hyperbolic CLAP Curvature Parameter Sweep Training Script
REM Run training sequentially for curvature parameter c from 0.1 to 1.0 (step 0.1)

setlocal enabledelayedexpansion

REM Set base parameters
set PARQUET_DIR=./Datasets/AudioSet/train_balanced
set NUM_SAMPLES=10000
set DEVICE=cuda:0
set EPOCHS=10
set BATCH_SIZE=16
set LEARNING_RATE=0.0001
set TEMPERATURE=1.0
set TRAIN_RATIO=0.8
set SEED=42

REM SwanLab parameters
set USE_SWANLAB=--use-swanlab
set SWANLAB_PROJECT=Hyperbolic_CLAP
set SWANLAB_WORKSPACE=Centauri

REM Base output directory
set BASE_OUTPUT_DIR=./checkpoints_hyperbolic

echo ============================================================
echo Starting Curvature Parameter Sweep Training
echo ============================================================
echo Training Parameters:
echo   Dataset Directory: %PARQUET_DIR%
echo   Number of Samples: %NUM_SAMPLES%
echo   Device: %DEVICE%
echo   Epochs: %EPOCHS%
echo   Batch Size: %BATCH_SIZE%
echo   Learning Rate: %LEARNING_RATE%
echo   Temperature: %TEMPERATURE%
echo   Train Ratio: %TRAIN_RATIO%
echo   Random Seed: %SEED%
echo   SwanLab: Enabled
echo   SwanLab Project: %SWANLAB_PROJECT%
echo   SwanLab Workspace: %SWANLAB_WORKSPACE%
echo ============================================================
echo.

REM Define curvature array
set CURVATURES=0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1

REM Counter
set COUNTER=0

REM Loop through each curvature value
for %%c in (%CURVATURES%) do (
    set /a COUNTER+=1
    
    REM Create output directory based on curvature (replace dot with underscore for path safety)
    set CURVATURE_STR=%%c
    set CURVATURE_STR=!CURVATURE_STR:.=_!
    set OUTPUT_DIR=!BASE_OUTPUT_DIR!_c!CURVATURE_STR!
    
    echo ============================================================
    echo Training !COUNTER!/10: Curvature Parameter c = %%c
    echo ============================================================
    echo Output Directory: !OUTPUT_DIR!
    echo Start Time: %date% %time%
    echo.
    
    REM Run training
    python train_hyperbolic_clap.py ^
        --parquet-dir "%PARQUET_DIR%" ^
        --num-samples %NUM_SAMPLES% ^
        --device %DEVICE% ^
        --output-dir "!OUTPUT_DIR!" ^
        --epochs %EPOCHS% ^
        --batch-size %BATCH_SIZE% ^
        --learning-rate %LEARNING_RATE% ^
        --c %%c ^
        --temperature %TEMPERATURE% ^
        --train-ratio %TRAIN_RATIO% ^
        --seed %SEED% ^
        %USE_SWANLAB% ^
        --swanlab-project %SWANLAB_PROJECT% ^
        --swanlab-workspace %SWANLAB_WORKSPACE%
    
    REM Check if training succeeded
    if errorlevel 1 (
        echo.
        echo [ERROR] Training failed for curvature c=%%c!
        echo End Time: %date% %time%
        echo Continuing to next curvature parameter...
        echo.
    ) else (
        echo.
        echo [SUCCESS] Training completed for curvature c=%%c!
        echo End Time: %date% %time%
        echo.
    )
    
    REM Wait 2 seconds to avoid output confusion
    timeout /t 2 /nobreak >nul
)

echo ============================================================
echo All Curvature Parameter Training Completed!
echo ============================================================
echo.
echo Training results saved in the following directories:
set COUNTER=0
for %%c in (%CURVATURES%) do (
    set /a COUNTER+=1
    set CURVATURE_STR=%%c
    set CURVATURE_STR=!CURVATURE_STR:.=_!
    echo   !COUNTER!. !BASE_OUTPUT_DIR!_c!CURVATURE_STR! ^(c=%%c^)
)
echo.
echo Each directory contains:
echo   - projection_epoch_X.pth: Checkpoint for each epoch
echo   - best_projection_epoch_X.pth: Best model
echo   - setup.json: Training record ^(contains all parameters and losses^)
echo.
echo Completion Time: %date% %time%
echo.

pause
