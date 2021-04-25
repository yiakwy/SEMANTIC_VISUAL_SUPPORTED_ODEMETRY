#/usr/bin/bash
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

BUILD_DEST="build"
DEST=${ROOT}/${BUILD_DEST}/proto_codec

mkdir -p ${DEST}

get_proto_files() {
  echo "$(find ${ROOT}/proto -iname "*.proto")"
}

function generate_cpp_proto()
{
	protoc -I${ROOT}/proto --cpp_out=${DEST} --proto_path=${DEST} $(get_proto_files)
}

function generate_python_proto()
{
	protoc -I${ROOT}/proto --python_out=${DEST} --proto_path=${DEST} $(get_proto_files)
}

help() {
  echo "sh scripts/gen_proto.sh --lang cpp"
}

main() {
  lang=
  while [[ "$#" -gt 0 ]]; do
	case $1 in 
		-l|--lang)
		       lang="$2"; shift ;;
		*) help
	esac
	shift
  done

  echo "FLAGS.lang : $lang"
  case $lang in 
	  cpp|CPP) generate_cpp_proto;;
	  python|Python|PYTHON) generate_python_proto;;
	  *) echo "$lang is not supported yet";;
  esac

}

echo "args: $@"
echo "$(get_proto_files)"
main $@
