# clang++ on the same AOTInductor wrapper (see gxx_fix.sh). modal run cuda_modal.py::debug --script scratch/clang_try.sh
exec > >(tee /root/.boltz/artifacts/clang_try.txt) 2>&1  # kept on the Volume if the client goes away
set -u
W=$(ls /root/.boltz/inductor-cache/cyl2l4t4rt7xekyyefxw3rt7xbg6bsqxtlqruigxxvbj32ecrosr/*.wrapper.cpp | head -1)
echo "#include <torch/csrc/inductor/aoti_include/cuda.h>" > /tmp/pch.h
COMMON="-D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D TORCH_INDUCTOR_PRECOMPILE_HEADERS -D C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX2 -D USE_MMAP_SELF -D USE_CUDA -O0 -DNDEBUG -fno-omit-frame-pointer -g1 -march=native -fPIC -std=c++20 -Wno-unused-variable -fopenmp -include /tmp/pch.h -I/usr/local/include/python3.12 -I/usr/local/lib/python3.12/site-packages/torch/include -I/usr/local/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda/include -mavx2 -mfma -mf16c -c"
echo "== clang++ $(clang++ --version | head -1)"
/usr/bin/time -f "rc=%x wall=%es maxrss=%MkB" clang++ "$W" $COMMON -fno-trapping-math -ffp-contract=off -Wno-unknown-pragmas -o /tmp/w_clang.o 2>&1 | grep -E "rc=|error|Stack dump|PLEASE" | head -8
ls -la /tmp/w_clang.o 2>&1
