set -e

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# import libraries
source $ROOT/scripts/utils.sh

SAVER="${ROOT}/log/"
OUTPUT="${ROOT}/log/video"

# if [ ! -d ${SAVER} ]; then
# info "Making directory ${SAVER}"
mkdir -p ${SAVER}
# fi

# if [ ! -d ${OUTPUT} ]; then
# info "Making directory ${OUTPUT}"
mkdir -p ${OUTPUT}
# fi

if dpkg-query -l ffmpeg > /dev/null 2>&1; then
  warn "ffmpeg not installed. installing ..."
  sudo apt-get install -y ffmpeg 
  info "complete installing ffmpeg."
fi 

task="user_test/No-1/liuwen"
date="202003-07"
seqid="000049.20200307084921992_R0403S00200012_2020030435REL"
cameraid="encpic_000001"

fn=${OUTPUT}/"user_test"

images_source_dir="${ROOT}/data/${task}/${date}/${seqid}/${cameraid}/"

function compose_video() {
  info "Composing ${fn}.mp4"
  if [ -f ${OUTPUT}/${fn}.mp4 ]; then
    rm ${fn}.mp4
  fi
  ffmpeg -r:v 15 -pattern_type glob -i "${images_source_dir}/*.jpeg" \
    -vcodec libx264 -pix_fmt yuv420p -crf 28 -an -n "${fn}.mp4"
  
}

compose_video
