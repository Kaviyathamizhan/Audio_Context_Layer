@echo off
setlocal enabledelayedexpansion

:: Audio Context Layer (ACL) Turnkey Execution Script for Windows
:: Usage:
::   scripts\run_all.bat [step] [data_dir] [author_name]
:: Steps:
::   clone_esc50, generate, test, train_crnn, train_ast, evaluate, edge, report, all

set STEP=%~1
if "%STEP%"=="" set STEP=all
set DATA_DIR=%~2
if "%DATA_DIR%"=="" set DATA_DIR=dataset
set AUTHOR=%~3
if "%AUTHOR%"=="" set AUTHOR=Researcher

echo ========================================================
echo Audio Context Layer Pipeline - Step: %STEP%
echo Dataset Dir: %DATA_DIR%
echo Author: %AUTHOR%
echo ========================================================

if "%STEP%"=="clone_esc50" goto DO_CLONE
if "%STEP%"=="generate" goto DO_GENERATE
if "%STEP%"=="test" goto DO_TEST
if "%STEP%"=="train_crnn" goto DO_TRAIN_CRNN
if "%STEP%"=="train_ast" goto DO_TRAIN_AST
if "%STEP%"=="evaluate" goto DO_EVALUATE
if "%STEP%"=="edge" goto DO_EDGE
if "%STEP%"=="report" goto DO_REPORT
if "%STEP%"=="all" goto DO_ALL

echo Unknown step '%STEP%'. Options: clone_esc50, generate, test, train_crnn, train_ast, evaluate, edge, report, all
exit /b 1

:DO_ALL
call :DO_CLONE
call :DO_GENERATE
call :DO_TEST
call :DO_TRAIN_AST
call :DO_TRAIN_CRNN
call :DO_EVALUATE
call :DO_EDGE
call :DO_REPORT
echo [SUCCESS] Entire ACL pipeline finished!
exit /b 0

:DO_CLONE
if not exist "ESC-50" (
    echo [1/8] Cloning ESC-50 dataset...
    git clone --depth 1 https://github.com/karolpiczak/ESC-50.git ESC-50
    rd /s /q ESC-50\.git 2>nul
) else (
    echo ESC-50 already present.
)
if "%STEP%"=="clone_esc50" exit /b 0
goto :eof

:DO_GENERATE
echo [2/8] Generating synthetic audio scenes and QA pairs...
python -m acl.data_gen --esc50 ESC-50 --out %DATA_DIR% --n_train 500 --n_val 80 --n_test 120 --seed 42
if errorlevel 1 exit /b 1
if "%STEP%"=="generate" exit /b 0
goto :eof

:DO_TEST
echo [3/8] Running internal consistency and leakage unit tests...
python -m pytest -q tests
if errorlevel 1 exit /b 1
if "%STEP%"=="test" exit /b 0
goto :eof

:DO_TRAIN_AST
echo [4/8] Training Main Model: AST (frozen AudioSet embeddings) + BiGRU head...
python -m acl.train_sed --data %DATA_DIR% --backend ast --out runs/ast --epochs 25 --bs 32 --seed 42
if errorlevel 1 exit /b 1
if "%STEP%"=="train_ast" exit /b 0
goto :eof

:DO_TRAIN_CRNN
echo [5/8] Training Baseline: Lightweight Edge CRNN...
python -m acl.train_sed --data %DATA_DIR% --backend crnn --out runs/crnn --epochs 30 --bs 32 --seed 42
if errorlevel 1 exit /b 1
if "%STEP%"=="train_crnn" exit /b 0
goto :eof

:DO_EVALUATE
echo [6/8] Evaluating Models (validation tuning, SED F1, QA accuracy, error breakdown)...
python -m acl.evaluate --data %DATA_DIR% --run runs/ast
python -m acl.evaluate --data %DATA_DIR% --run runs/crnn
if errorlevel 1 exit /b 1
if "%STEP%"=="evaluate" exit /b 0
goto :eof

:DO_EDGE
echo [7/8] Measuring Edge Feasibility (FP32 latency, size, dynamic INT8 quantization)...
python -m acl.edge_report --data %DATA_DIR% --run runs/ast
python -m acl.edge_report --data %DATA_DIR% --run runs/crnn
if errorlevel 1 exit /b 1
if "%STEP%"=="edge" exit /b 0
goto :eof

:DO_REPORT
echo [8/8] Compiling Technical PDF Report...
python -m acl.make_report --data %DATA_DIR% --runs runs/ast runs/crnn --author "%AUTHOR%" --out report/ACL_technical_report.pdf
if errorlevel 1 exit /b 1
echo [REPORT CREATED] Output located at report/ACL_technical_report.pdf
if "%STEP%"=="report" exit /b 0
goto :eof
