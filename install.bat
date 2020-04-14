@echo off

set argC=0
for %%x in (%*) do Set /A argC+=1

if %argC% NEQ 2 (
	echo "ERROR! Illegal number of parameters (%argC%) Usage: install.bat conda_install_path" environment_name
    goto:eof
)

set conda_install_path=%1
set conda_env_name=%2

echo Conda install path: %conda_install_path%

set PATH=%conda_install_path%\condabin;%PATH%

echo ****************** Creating conda environment %conda_env_name% python=3.7 ******************
call conda create -y --name %conda_env_name%

echo.
echo.
echo ****************** Activating conda environment %conda_env_name% ******************
call activate %conda_env_name%

echo.
echo.
echo ****************** Installing pytorch with cuda9 ******************
call conda install -y pytorch torchvision cudatoolkit=9.0 -c pytorch 

echo.
echo.
echo ****************** Installing matplotlib 2.2.2 ******************
call conda install -y matplotlib=2.2.2

echo.
echo.
echo ****************** Installing pandas ******************
call conda install -y pandas

echo.
echo.
echo ****************** Installing opencv ******************
call pip install opencv-python

echo.
echo.
echo ****************** Installing tensorboardX ******************
call pip install tensorboardX

echo.
echo.
echo ****************** Installing cython ******************
call conda install -y cython

echo.
echo.
echo ****************** Installing coco toolkit ******************
call pip install pycocotools

echo.
echo.
echo ****************** Installing jpeg4py python wrapper ******************
call pip install jpeg4py 

echo.
echo.
echo ****************** Preparing directory for networks ******************
mkdir pytracking\networks

echo.
echo.
echo ****************** Setting up environment ******************
call python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
call python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"

goto:label1
echo.
echo.
echo ****************** Installing jpeg4py ******************
while true; do
    read -p "Install jpeg4py for reading images? This step required sudo privilege. Installing jpeg4py is optional, however recommended. [y,n]  " install_flag
    case $install_flag in
        [Yy]* ) sudo apt-get install libturbojpeg; break;;
        [Nn]* ) echo "Skipping jpeg4py installation!"; break;;
        * ) echo "Please answer y or n  ";;
    esac
done
:label1

echo.
echo.
echo ****************** Installation complete! ******************
