conda env create -f conda_setup/ai_cjs.yml

echo 'Created ai-cjs-env'

eval "$(conda shell.bash hook)"
conda activate ai-cjs-env

echo 'Activated ai-cjs-env'

# conda install -c anaconda pywin32  # Enable this line for windows support

echo 'Building postcodes_io'

conda skeleton pypi postcodes_io
conda-build postcodes_io --output-folder conda_setup/pypi_builds

echo 'Built postcodes_io'

echo 'Building osgridconverter'

conda skeleton pypi osgridconverter
conda-build osgridconverter --output-folder conda_setup/pypi_builds

echo 'Built osgridconverter'

conda index conda_setup/pypi_builds

WORKING_DIR=$(pwd)

conda install -y -c file://"$WORKING_DIR"/conda_setup/pypi_builds/ postcodes_io
conda install -y -c file://"$WORKING_DIR"/conda_setup/pypi_builds/ osgridconverter

echo "Installed postcodes_io and osgridconverter from pypi"

# clear old versions
rm -rf conda_setup/ai_cjs_build/linux-64/ conda_setup/ai_cjs_build/noarch/ conda_setup/ai_cjs_build/icons/
rm -f conda_setup/ai_cjs_build/index.html conda_setup/ai_cjs_build/channeldata.json
conda-build conda_setup/ai_cjs_build/ --output-folder conda_setup/ai_cjs_build
conda index conda_setup/ai_cjs_build
conda install -y -c file://"$WORKING_DIR"/conda_setup/ai_cjs_build/ ai_cjs

if [ $? -eq 0 ]; then echo "[ai_cjs installed]"; else echo "[ai_cjs install Failed]"; exit 1; fi

# get the install location for AI_CJS
DEL="bin*python"
PYTHONPATH="$(which python)"
ENVLOC="${PYTHONPATH/$DEL}lib/python3.7/site-packages/ai_cjs/"

cp ai_cjs/who_am_i.txt "$ENVLOC"

conda build purge
