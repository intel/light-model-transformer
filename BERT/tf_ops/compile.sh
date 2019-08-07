PYTHON='python'
TF_CFLAGS=`$PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'`
TF_LFLAGS=`$PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'`

#icpc -std=c++11 -shared bert.cc -o bert.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -qopenmp -lmkl_rt -liomp5 -xCORE-AVX512
g++ -std=c++11 -shared bert.cc -o bert.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -fopenmp -lmkl_rt -liomp5 -xCORE-AVX512
