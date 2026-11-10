# Replay the AOTInductor wrapper compile that crashes g++ (section 7) under gdb.
# Runs in the CPU debug container: modal run cuda_modal.py::debug --script scratch/gxx_gdb.sh
set -u
W=$(ls /root/.boltz/inductor-cache/cyl2l4t4rt7xekyyefxw3rt7xbg6bsqxtlqruigxxvbj32ecrosr/*.wrapper.cpp | head -1)
echo "wrapper: $W"; ls -la "$W"; wc -l "$W"
echo "#include <torch/csrc/inductor/aoti_include/cuda.h>" > /tmp/pch.h  # what Inductor precompiles
echo "stack soft/hard: $(ulimit -s) / $(ulimit -Hs); cpus $(nproc)"; g++ --version | head -1
echo "biggest functions (lines):"
awk '/^[A-Za-z].*\) *\{$|^[A-Za-z].*\{$/{name=$0; start=NR} /^\}/{if (name!="") print NR-start, substr(name,1,120); name=""}' "$W" | sort -rn | head -5

FLAGS="-D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D TORCH_INDUCTOR_PRECOMPILE_HEADERS -D C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX2 -D USE_MMAP_SELF -D USE_CUDA -O0 -DNDEBUG -fno-omit-frame-pointer -g1 -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-finite-math-only -fno-unsafe-math-optimizations -fmath-errno -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -march=native -fPIC -Wall -std=c++20 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp -include /tmp/pch.h -I/usr/local/include/python3.12 -I/usr/local/lib/python3.12/site-packages/torch/include -I/usr/local/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda/include -mavx2 -mfma -mf16c -c"

echo; echo "== 1. the failing command as Inductor ran it (-O0)"
/usr/bin/time -f "rc=%x wall=%es maxrss=%MkB" g++ "$W" $FLAGS -o /tmp/w.o 2>&1 | grep -vE "warning: integer constant|AOTI_TORCH_ERROR_CODE_CHECK\(aoti_torch_cuda_randint|^\s+\||^\s+[0-9]+ \|" | tail -6

echo; echo "== 2. cc1plus under gdb"
CC1=$(g++ "$W" $FLAGS -o /tmp/w.o -### 2>&1 | grep -m1 cc1plus | sed 's/"//g')
echo "${CC1:0:300} ..."
gdb -q -batch -ex "run" -ex "p \$_siginfo._sifields._sigfault.si_addr" -ex "p \$sp" \
    -ex "info proc mappings" -ex "bt 12" -ex "bt -6" --args $CC1 2>&1 \
  | grep -vE "warning: integer constant|randint_low_out|^\s+\||^\s+[0-9]+ \||^\[Thread|^\[New|^\[Inferior" \
  | grep -E "SIGSEGV|si_addr|^\\\$|\[stack\]|^#|0x[0-9a-f]+ +0x[0-9a-f]+ .*stack" | head -40
