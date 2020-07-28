echo "Installing ai_cjs and required dependencies."

echo "A top data directory is required. Please input absolute path of this directory. "
echo "This path should be in a separate location and will be used to store the generated data."
echo "It should end in '.../ai_cjs/'  e.g. '/home/user/Documents/Scratch/ai_cjs/' "

read -r SCRATCHPATH
{ # try
  cd "$SCRATCHPATH"
  cd -
} || { #catch
  echo "File not found"
  exit
}

touch ./ai_cjs/who_am_i.txt
echo "#This file is hidden from git it needs to be present for config to pickup the location of scratch" >>./ai_cjs/who_am_i.txt
echo "$SCRATCHPATH" >>./ai_cjs/who_am_i.txt

echo "A Conda environment will now be created called 'ai-cjs-env' it will overwrite any preexisting environment with
the same name. The build will fail if you do not have an SSH key setup with github."
read -r -p "Continue? [y/N] " RESPONSE
if [[ "$RESPONSE" =~ ^([yY][eE][sS]|[yY])$ ]]; then
  bash ./conda_setup/install.sh
else
  exit
fi


