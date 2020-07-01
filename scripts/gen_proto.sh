#/usr/bin/bash
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

BUILD_DEST="build"
DEST=${ROOT}/${BUILD_DEST}/proto_codec

mkdir -p ${DEST}


function genreate_cpp_proto()
{
  protoc -I${ROOT}/proto --cpp_out=${DEST} ${ROOT}/proto/*.proto
}

function generate_python_proto()
{
  protoc -I${ROOT}/proto --python_out=${DEST} ${ROOT}/proto/*.proto
}

generate_python_proto
