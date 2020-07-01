set -e
git clone https://github.com/google/googletest ~/WorkSpace/Github/googletest
cd ~/WorkSpace/Github/googletest
mkdir -p build
cd build
cmake ..
make 
sudo make install
