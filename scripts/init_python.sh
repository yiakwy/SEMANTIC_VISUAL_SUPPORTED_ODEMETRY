ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

CONDA_BASE=$( conda info --base )
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py36
pip install -r ${ROOT}/python/requirements.txt
