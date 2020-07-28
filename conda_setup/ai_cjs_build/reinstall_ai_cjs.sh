# clear old versions
rm -rf conda_setup/ai_cjs_build/linux-64/ conda_setup/ai_cjs_build/noarch/ conda_setup/ai_cjs_build/icons/
rm -f conda_setup/ai_cjs_build/index.html conda_setup/ai_cjs_build/channeldata.json conda_setup/ai_cjs_build/linux-64/ai_cjs*.tar.bz2

# download version specified in conda_setup/ai_cjs_build/meta.yaml
eval "$(conda shell.bash hook)"
conda activate ai-cjs-env
conda uninstall -y ai_cjs
conda-build conda_setup/ai_cjs_build/ --output-folder conda_setup/ai_cjs_build
conda index conda_setup/ai_cjs_build
WORKING_DIR=$(pwd)
conda install -y -c file://"$WORKING_DIR"/conda_setup/ai_cjs_build/ ai_cjs