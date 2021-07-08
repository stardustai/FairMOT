pip install cython
pip install -r requirements.txt
git clone -b pytorch_1.7 https://github.com/ifzhang/DCNv2.git
cd DCNv2
./make.sh
ln -s ~/data/kitti_tracking dataset/
ln -s ~/data/MOT20 dataset/
ln -s ~/data/DETRAC/ dataset/