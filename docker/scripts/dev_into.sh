ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

set -e

# include libraries
source $ROOT/scripts/utils.sh

DOCKER_REPO=openhdmap/study/svso
BASE_IMG=$DOCKER_REPO:dev-x86_64-20200804_0134_base
STAGE_1=$DOCKER_REPO:dev-x86_64-20200806_1908_stage_1

IMG=

# light wrapper for docker to create an image
_run_container() {
local container_id=$1
local img=$2

USER_ID=$(id -u)
GRP=$(id -g -n)
GRP_ID=$(id -g)
DOCKER_HOME="/home/$USER"

# see https://docs.docker.com/engine/reference/run/
# see https://github.com/ApolloAuto/apollo/blob/v1.5.0/docker/scripts/dev_start.sh to add more devices (gpu devices already handled)
# you can use nvidia-smi to test the envrionment
docker run --gpus all -it \
	-d \
	--privileged \
	--name="$container_id" \
	-e DOCKER_USER=$USER \
	-e USER=$USER \
	-e DOCKER_USER_ID=$USER_ID \
	-e DOCKER_GRP=$GRP \
	-e DOCKER_GRP_ID=$GRP_ID \
	-e /tmp/.X11-unix:/tmp/.X11-unix:rw \
	-v $ROOT:/tmp/deploy/SEMANTIC_VISUAL_SUPPORTED_ODEMETRY \
	-v /media:/media \
	-v $HOME/.cache:${DOCKER_HOME}/.cache \
	-v /etc/localtime:/etc/localtime:ro \
	--net host \
	--add-host in_dev_docker:127.0.0.1 \
	--add-host `hostname`:127.0.0.1 \
	--hostname in_dev_docker \
	--shm-size 2048M \
	$img /bin/bash

docker container start $container_id
# add user
# https://docs.docker.com/engine/security/rootless/
docker exec $container_id bash -c 'docker/scripts/add_user.sh'
}

_into_container() {
local container_id=$1
args=(
)
xhost +local:root 1>/dev/null 2>&1
docker exec -it -u $USER $container_id /bin/bash
xhost -local:root 1>/dev/null 2>&1
}

run_contianer() {
img_type=$1
case $img_type in 
	base)
	CONTAINER_ID="base_env"
	IMG=$BASE_IMG
	_run_container $CONTAINER_ID $IMG
		;;
	base_py36)
	CONTAINER_ID="base_py36_env"
	IMG=$STAGE_1
	_run_container $CONTAINER_ID $IMG
		;;
	svso:lastest)
		# @todo TODO
		;;
	*)
	 	err "Not supported options: $@; see help. Pull reqeusts are welcome!"
	 	;;
esac
}

into_container() {
local img_type=$1
case $img_type in 
	base)
		CONTAINER_ID="base_env"
		_into_container $CONTAINER_ID
		;;
	base_py36)
		CONTAINER_ID="base_py36_env"
		_into_container $CONTAINER_ID
		;;
	svso:lastest)
		# @todo TODO
		;;
	*)
		err "Not supported options: $@; see help. Pull reqeusts are welcome!"
		;;
esac
}

main() {

sub_cmd=$1
echo "sub command : <$sub_cmd>"
case $sub_cmd in 
	start)
		run_contianer "${@:2}"
		;;
	into)
		into_container "${@:2}"
		;;
	*)
	 	err "Not supported command <$sub_cmd>; see help. Pull requests are welcome!"
	 	;;
esac

}

echo "args: $@"
main "$@"
