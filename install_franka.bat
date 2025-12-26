@echo off
REM 安装 FrankaPickPlace 包 - 完整版

echo ================================================
echo 步骤 1: 清理 Python 缓存
echo ================================================
cd /d C:\Allfile\WS\RL-Franka-PickPlace
if exist source\FrankaPickPlace\__pycache__ rmdir /s /q source\FrankaPickPlace\__pycache__
if exist source\FrankaPickPlace\tasks\__pycache__ rmdir /s /q source\FrankaPickPlace\tasks\__pycache__
if exist source\FrankaPickPlace\tasks\manager_based\__pycache__ rmdir /s /q source\FrankaPickPlace\tasks\manager_based\__pycache__
if exist source\FrankaPickPlace\tasks\manager_based\frankapickplace\__pycache__ rmdir /s /q source\FrankaPickPlace\tasks\manager_based\frankapickplace\__pycache__
if exist source\FrankaPickPlace\tasks\manager_based\frankapickplace\mdp\__pycache__ rmdir /s /q source\FrankaPickPlace\tasks\manager_based\frankapickplace\mdp\__pycache__

echo ================================================
echo 步骤 2: 卸载旧版本（如果有）
echo ================================================
C:\IsaacSim\python.bat -m pip uninstall FrankaPickPlace -y 2>nul

echo ================================================
echo 步骤 3: 安装当前目录的 FrankaPickPlace 包
echo ================================================
C:\IsaacSim\python.bat -m pip install -e source/FrankaPickPlace --no-deps

echo ================================================
echo 步骤 4: 验证安装
echo ================================================
REM 使用 Isaac Lab 的 Python 来验证（启动器会设置正确的环境）
C:\IsaacSim\python.bat -c "import FrankaPickPlace; print('导入路径:', FrankaPickPlace.__file__); print('版本:', getattr(FrankaPickPlace, '__version__', '未知'))"

echo ================================================
echo 安装完成！
echo ================================================
echo 请确保删除任何旧的 PickAndPlace 目录或将其移出 Python 路径
pause
