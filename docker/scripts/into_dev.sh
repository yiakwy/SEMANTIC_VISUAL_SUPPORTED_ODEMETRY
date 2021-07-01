#/usr/bin/bash
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

set -e

# include libraries
source $ROOT/scripts/utils.sh

# predefined reposiotry and images
DOCKER_REPO=openhdmap/study/svso
BASE_IMG=$DOCKER_REPO:dev-x86_64-20210701_1749_base
STAGE_1=$DOCKER_REPO:dev-x86_64-20200806_1908_stage_1

# IMAGE used to update
IMG=

# used to update from image registry
VERSION=

# created containers
containers=

## svso docker utilties

get_containers() {
  containers=$(docker container ls -a --format "{{.Names}}")
  echo "created containers : "
  for c in ${containers[@]}; do
   echo "CONTAINER: $c"
  done
}

# light wrapper for docker to create an image
_run_container() {
  local container_id=$1
  local img=$2
  local is_container_created=

  USER_ID=$(id -u) # 0 for root user
  GRP=$(id -g -n) 
  GRP_ID=$(id -g) # 0 for root user
  DOCKER_HOME="/home/$USER"

  # see https://github.com/thewtex/docker-opengl-mesa/blob/master/run.sh to add display
  DISPLAY=$DISPLAY
  # @todo TODO check DISPLAY

  # ADD HOST NAME
  HOSTNAME="in_docker_dev"

  # RUNTIME
  RUNTIME="nvidia" # "runc"

  # SHARE Memory Mode: just for development purpose, not safe for hosting public services
  IPC_NAMESPACE=host

  # Enable GUI Application
  X11_unix=/tmp.X11-unix

  # check if the container has already been booted
  get_containers

  is_container_created=false  
  for c in ${containers[@]}; do
    if [[ $c -eq "$container_id" ]]; then
      is_container_created=true
      break
    fi
  done

  if [ ! $is_container_created ]; then
    echo "creating container $container_id"
    # see https://docs.docker.com/engine/reference/run/
    # see https://github.com/ApolloAuto/apollo/blob/v1.5.0/docker/scripts/dev_start.sh to add more devices (gpu devices already handled)
    # addtional useful things:
    #  1. ssh forward inside docker : https://gist.github.com/d11wtq/8699521
    # you can use nvidia-smi to test the envrionment
    set -x
    docker run --gpus all -it \
      -d \
      --privileged \
      --name="$container_id" \
      -e CONTAINER_NAME=$container_id \
      -e DOCKER_USER=$USER \
      -e USER=$USER \
      -e DOCKER_USER_ID=$USER_ID \
      -e DOCKER_GRP=$GRP \
      -e DOCKER_GRP_ID=$GRP_ID \
      -e /tmp/.X11-unix:/tmp/.X11-unix:rw \
      -e DISPLAY=$DISPLAY \
      -e SVSO_INSTALL='/opt/deploy/svso' \
      -e LC_ALL='' \
      -e SSH_AUTH_SOCK=$SSH_AUTH_SOCK \
      -e XSOCK=/tmp/.X11-unix \
      -u $USER_ID:$GRP_ID \
      -v /media:/media \
      -v $HOME/.cache:${DOCKER_HOME}/.cache \
      -v /etc/localtime:/etc/localtime:ro \
      -v /etc/timezone:/etc/timezone:ro \
      -v /etc/resolv.conf:/etc/resolv.conf:ro \
      -v /etc/hosts:/etc/hosts:ro \
      -v /etc/passwd:/etc/passwd:ro \
      -v /etc/group:/etc/group:ro \
      -v /etc/sudoers.d:/etc/sudoers.d:ro \
      -v /etc/sudoers:/etc/sudoers:ro \
      -v $X11_unix:$X11_unix:rw \
      -v /run/user/$USER_ID/keyring/ssh:/ssh-agent:rw \
      -v /home/$USER:/home/$USER:rw \
      -v /dev/snd:/dev/snd:rw \
      --ipc=$IPC_NAMESPACE \
      --net host \
      --add-host $HOSTNAME:127.0.0.1 \
      --add-host `hostname`:127.0.0.1 \
      --hostname $HOSTNAME \
      --group-add sudo \
      --group-add audio \
      --device /dev/dri/card1 \
      --workdir "/home/$USER" \
      --runtime=$RUNTIME \
      $img /bin/bash
    set +x 
  fi
 
  echo "starting container $container_id"
  docker container start -a -i $container_id
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
    CONTAINER_ID="svso_base_env"
    IMG=$BASE_IMG
    _run_container $CONTAINER_ID $IMG
      ;;
    base_py36)
    CONTAINER_ID="svso_base_py36_env"
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
      CONTAINER_ID="svso_base_env"
      _into_container $CONTAINER_ID
      ;;
    base_py36)
      CONTAINER_ID="svso_base_py36_env"
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
