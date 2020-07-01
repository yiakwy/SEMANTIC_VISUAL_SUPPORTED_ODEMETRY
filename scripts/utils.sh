COLOR_OFF='\033[0m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'

function info() {
	(echo >&2 -e "${GREEN} [INFO] ${COLOR_OFF} $*")
}

function err() {
	(echo >&2 -e "${RED} [ERROR] ${COLOR_OFF} $*" )
}

function warn() {
	(echo >&2 -e "${YELLOW} [WARNING] ${COLOR_OFF} $*" )
}

