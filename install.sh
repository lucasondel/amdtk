#!/usr/bin/env bash 

#
# Install the dependencies needed to run AMDTK.
# 

# We avoid symbolic links as it may confuses the installation process.
amdtk_root=$(pwd -P)

anaconda_path="$HOME/anaconda3"
env_name="py35_amdtk"

while getopts ":e:hp:" opt; do
    case $opt in
        e) 
            env_name=$OPTARG
            ;;
        p)
            anaconda_path=$OPTARG
            ;;
        h)
            echo "usage: $0 [OPTIONS]                                         "
            echo "                                                            "
            echo "Install the dependencies needed to run AMDTK recipes.       "
            echo "                                                            "
            echo "Options:                                                    "
            echo "  -e NAME     Anaconda environment name. Default: $env_name."
            echo "  -h          Show this message.                            "
            echo "  -p PATH     Path to an existing Anaconda distribution.    "
            echo "              Default: $anaconda_path.                      "
            exit 0
            ;;
        \?) 
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done

if [ ! -e "$anaconda_path/bin/python" ]; then
    echo "Invalid anaconda path. Exiting the installation script."
    exit 1
fi

# Directory where we will install the external tools.
mkdir -p $amdtk_root/tools

# Set the path to the correct python distribution for the the rest of the 
# installation.
unset PYTHONPATH
export PATH=$anaconda_path/bin:$PATH

# We create a specific environment for AMDTK if it doesn't exists.
echo "Installing anaconda environment \"$env_name\"... "
if [ -z "`conda env list | grep $env_name`" ]; then
    conda env create --name $env_name python=3 -f "$amdtk_root/py35_amdtk.yml"
fi || exit 1

# Activate the new environment to install other dependencies.
source activate $env_name

# Install sselogsumexp.
echo "Installing sselogsumexp... "
if [ ! -e "$amdtk_root/tools/logsumexp" ]; then
    cd "$amdtk_root/tools"
    git clone "https://github.com/rmcgibbo/logsumexp.git" 
    cd logsumexp
    python setup.py install 
    cd "$amdtk_root"
fi || exit 1

# Download and install sph2pipe. This is needed for features extraction.
echo "Installing sph2pipe... "
if [ ! -e $amdtk_root/tools/sph2pipe_v2.5.tar.gz ]; then    
    wget -P $amdtk_root/tools "https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/ctools/sph2pipe_v2.5.tar.gz" --no-check-certificate 
fi || exit 1

if [ ! -e $amdtk_root/tools/sph2pipe_v2.5 ]; then
    cd "$amdtk_root/tools"
    tar xvf sph2pipe_v2.5.tar.gz
    cd $amdtk_root/tools/sph2pipe_v2.5
    gcc -o sph2pipe *.c -lm 
    cd ../../
fi || exit 1


echo "Creating path file... "
new_path="$anaconda_path/bin:$amdtk_root/scripts:$amdtk_root/tools/sph2pipe_v2.5:\$PATH"
# Create the path.sh file to use the newly created environment.
echo "# Setting python environment.            " >  "$amdtk_root/tools/path.sh"
echo "unset PYTHONPATH                         " >> "$amdtk_root/tools/path.sh"
echo "export PYTHONPATH=$amdtk_root            " >> "$amdtk_root/tools/path.sh"
echo "                                         " >> "$amdtk_root/tools/path.sh"
echo "# Add extra tools to the PATH.           " >> "$amdtk_root/tools/path.sh"
echo "export PATH=$new_path:$PATH              " >> "$amdtk_root/tools/path.sh"
echo "                                         " >> "$amdtk_root/tools/path.sh"
echo "# Selecting the AMDTK environment.       " >> "$amdtk_root/tools/path.sh"
echo "source activate $env_name                " >> "$amdtk_root/tools/path.sh"
echo "                                         " >> "$amdtk_root/tools/path.sh"
echo "# Disable multithreading.                " >> "$amdtk_root/tools/path.sh"
echo "export OPENBLAS_NUM_THREADS=1            " >> "$amdtk_root/tools/path.sh"
echo "export OMP_NUM_THREDS=1                  " >> "$amdtk_root/tools/path.sh" 
echo "export MKL_NUM_THREADS=1                 " >> "$amdtk_root/tools/path.sh"

# Install the recipes.
echo "Installing recipes... "
# timit recipe
cp "$amdtk_root/tools/path.sh" "$amdtk_root/recipes/timit" || exit 1
# wsj recipe
cp "$amdtk_root/tools/path.sh" "$amdtk_root/recipes/wsj" || exit 1
ln -fs "$amdtk_root/recipes/timit/utils" "$amdtk_root/recipes/wsj/utils" || exit 1
# babel turkish recipe
cp "$amdtk_root/tools/path.sh" "$amdtk_root/recipes/babel_turkish_clsp" || exit 1
ln -fs "$amdtk_root/recipes/timit/utils" "$amdtk_root/recipes/babel_turkish_clsp/utils" || exit 1
# WSJ no punctuation recipe
cp "$amdtk_root/tools/path.sh" "$amdtk_root/recipes/wsj_no_punc" || exit 1
ln -fs "$amdtk_root/recipes/timit/utils" "$amdtk_root/recipes/wsj_no_punc/utils" || exit 1
# zerocost recipe
cp "$amdtk_root/tools/path.sh" "$amdtk_root/recipes/zerocost" || exit 1
ln -fs "$amdtk_root/recipes/timit/utils" "$amdtk_root/recipes/zerocost/utils" || exit 1
