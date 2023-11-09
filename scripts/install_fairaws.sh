
#  The script is installing seamless_communication (internal) + fairseq2 on AWS cluster.

set -e
set -x

echo "Installing Conda"
export TGT=`echo ~/seacom_aws_dev`
rm -rf $TGT
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -qO /tmp/conda.sh
bash /tmp/conda.sh -bp $TGT
export CONDA=$TGT/bin/conda
export CONDA_ACTIVATE=$TGT/bin/activate
export ENV_N=sc_fr2_dev
echo "Next step will take ~15 minutes. Get some coffee" 
$CONDA create -y -n ${ENV_N} python=3.10 pytorch=2.0.1 pytorch-cuda=11.8 torchvision torchaudio \
             compilers libsndfile==1.0.31 gcc==11.4.0 \
    --strict-channel-priority --override-channels \
    -c https://aws-ml-conda.s3.us-west-2.amazonaws.com \
    -c pytorch \
    -c nvidia \
    -c conda-forge

echo "Setting LD_LIBRARY_PATH"
. $CONDA_ACTIVATE activate ${ENV_N}
if [ -z "$CONDA_PREFIX" ]; then 
  echo "CONDA_PREFIX env var is not set!" 
  exit 1
else 
   path=$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
   echo  "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH\n"  >> ${path}
fi
. $CONDA_ACTIVATE activate ${ENV_N}  # update env vars


#  NOTICE: to compile CUDA kernels, you need NVCC. On AWS cluster an easy way would be to get a GPU container:
#  srun -N 1 --gres=gpu:1 --cpus-per-task=20 --partition seamless --time 2400 --pty /bin/bash -l

#  Installing fairseq2.
echo "Installing fairseq2"
set -e
rm -rf fairseq2  # wipe existing clones
if [[ "${I_DONT_PLAN_TO_HACK_FAIRSEQ2:-No}" == "Yes" ]] ; then
pip install fairseq2 \
  --pre --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/nightly/pt2.0.1/cu118
else
nvidia-smi || echo "to compile CUDA kernels, you need NVCC.\n \
   On AWS cluster an easy way would be to get a GPU container.\n \
   Run smth like 'srun -N 1 --gres=gpu:1 --cpus-per-task=20 --partition seamless --time 2400 --pty /bin/bash -l' \n \
   and continue from "Installing fairseq2" line. \
   Terminating for now."
nvidia-smi || exit 1
cd $TGT
. $CONDA_ACTIVATE activate ${ENV_N}
git clone --recurse-submodules  git@github.com:facebookresearch/fairseq2.git
pip install -r fairseq2/fairseq2n/python/requirements-build.txt
cd fairseq2
pip install -e .  # it will install public fairseq2n, we rewrite it below
cd fairseq2n
args="-GNinja\
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80-real;80-virtual\
  -DFAIRSEQ2N_INSTALL_STANDALONE=ON\
  -DFAIRSEQ2N_PERFORM_LTO=ON\
  -DFAIRSEQ2N_TREAT_WARNINGS_AS_ERRORS=OFF\
  -DFAIRSEQ2N_USE_CUDA=ON\
  -DFAIRSEQ2N_BUILD_PYTHON_BINDINGS=ON\
  -DFAIRSEQ2N_PYTHON_DEVEL=OFF"
cmake ${args} -B build
cmake --build build
cd python && pip install .
fi
# Quick test
python -c "from fairseq2n.bindings.data.string import CString as CString"

echo "Installing seamless_communication"
cd $TGT
git clone git@github.com:fairinternal/seamless_communication.git
cd seamless_communication
pip install -e .   # editable mode for hacking


echo "One more time re-install fairseq2n (most propably overriden by seamless_communication)"
cd $TGT/fairseq2/fairseq2n/python
pip install .


echo "Finished."
echo "To activate the environment run: . $CONDA_ACTIVATE activate ${ENV_N}"
echo "Location of seamless_communication checkout: $TGT/seamless_communication"