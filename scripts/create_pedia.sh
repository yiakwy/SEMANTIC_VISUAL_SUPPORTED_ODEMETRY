ROOT="$( cd "$( dirname "$BASH_SOURCE[0]" )/.." && pwd )"

source "$ROOT/scripts/utils.sh"

info "creating pedia ..."
mkdir -p $ROOT/pedia
mkdir -p $ROOT/pedia/assets
mkdir -p $ROOT/pedia/assets/images

ln -s $ROOT/scripts/book.json $ROOT/pedia/book.json
# ln -s $ROOT/scripts/.gitbook.yaml $ROOT/pedia/.gitbook.yaml
cp $ROOT/README.md $ROOT/pedia
if [ -d $ROOT/pedia/docs ]; then
  rm -r $ROOT/pedia/docs
fi
cp -r $ROOT/docs $ROOT/pedia
cd $ROOT/pedia

# install pedia plugins
# gitbook install

#
gitbook init

# should already installed by scripts/install.sh
gitbook build ./ --output=./_book --log=info --debug
gitbook serve
