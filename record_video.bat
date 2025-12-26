@echo off
REM 运行训练好的策略进行可视化 - v2版本

cd /d %~dp0

echo 查找最新的checkpoint...

REM 设置日志目录路径
set "log_dir=%CD%\logs\rsl_rl\franka_pick_place"

REM 检查目录是否存在
if not exist "%log_dir%" (
    echo 错误: 训练目录不存在 %log_dir%
    echo 请先运行 train_franka.bat 进行训练
    pause
    exit /b 1
)

REM 查找最新修改的目录
set "latest_dir="
for /f "delims=" %%i in ('dir /b /ad /o-d "%log_dir%" 2^>nul') do (
    set "latest_dir=%%i"
    goto :found_dir
)

:found_dir
if not defined latest_dir (
    echo 错误: 在 %log_dir% 中找不到训练目录
    pause
    exit /b 1
)

echo 找到最新训练目录: %latest_dir%
set "checkpoint_dir=%log_dir%\%latest_dir%"

REM 查找最新的模型文件
set "latest_model="
for /f "delims=" %%i in ('dir /b /o-n "%checkpoint_dir%\model_*.pt" 2^>nul') do (
    set "latest_model=%%i"
    goto :found_model
)

:found_model
if not defined latest_model (
    echo 错误: 在 %checkpoint_dir% 中找不到模型文件
    pause
    exit /b 1
)

echo 使用模型: %latest_model%

echo.
echo 开始录制视频（无头模式）...
"C:\IsaacSim\python.bat" scripts/rsl_rl/play.py --task Template-Franka-Lift --num_envs 1 --load_run %latest_dir% --video --video_length 100 --headless
echo.
echo ============================================
echo 视频录制完成！
echo 查看: logs\rsl_rl\franka_pick_place\%run_dir%\videos\play\
echo ============================================

pause
