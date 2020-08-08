ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

set -e

# loading libraries
source "${ROOT}/scripts/utils.sh"

# install dependencies
pip install -r ${ROOT}/python/requirements.txt

# build external deep learning libraries
# build coco
make install -C $ROOT/vendors/github.com/coco/PythonAPI

# download coco models
model_file="mask_rcnn_coco.h5"
mkdir -p $ROOT/data/models/coco
if [ ! -f $ROOT/data/models/coco/$model_file ]; then
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/${model_file} -P $ROOT/data/models/coco
fi
ls -h $ROOT/data/models/coco

