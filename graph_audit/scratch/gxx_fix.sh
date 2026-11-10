# Follow-up to gxx_gdb.sh: how deep the wrapper nests, and two ways around the 8 MiB stack.
exec > >(tee /root/.boltz/artifacts/gxx_fix.txt) 2>&1  # kept on the Volume if the client goes away
# modal run cuda_modal.py::debug --script scratch/gxx_fix.sh
set -u
W=$(ls /root/.boltz/inductor-cache/cyl2l4t4rt7xekyyefxw3rt7xbg6bsqxtlqruigxxvbj32ecrosr/*.wrapper.cpp | head -1)
echo "#include <torch/csrc/inductor/aoti_include/cuda.h>" > /tmp/pch.h
python3 - "$W" <<'EOF'
import re, sys
depth = peak = 0; peak_line = 0; opened = []
for i, line in enumerate(open(sys.argv[1]), 1):
    code = re.sub(r'"(\\.|[^"\\])*"', '""', line.split("//")[0])
    for ch in code:
        if ch == "{":
            depth += 1; opened.append(i)
            if depth > peak: peak, peak_line = depth, i
        elif ch == "}":
            depth -= 1; opened.pop()
print(f"max brace depth {peak} at line {peak_line}")
lines = open(sys.argv[1]).read().splitlines()
for n in (peak_line - 6, peak_line - 3):
    print(f"{n}: {lines[n - 1][:160]}")
starts = [l for l in lines if l.strip().startswith(("if (", "while (", "for (", "do {"))]
print("control statements:", len(starts), "| sample:", [s.strip()[:80] for s in starts[:3]])
EOF

COMMON="-D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D TORCH_INDUCTOR_PRECOMPILE_HEADERS -D C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX2 -D USE_MMAP_SELF -D USE_CUDA -O0 -DNDEBUG -fno-omit-frame-pointer -g1 -march=native -fPIC -std=c++20 -Wno-unused-variable -fopenmp -include /tmp/pch.h -I/usr/local/include/python3.12 -I/usr/local/lib/python3.12/site-packages/torch/include -I/usr/local/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -I/usr/local/cuda/include -mavx2 -mfma -mf16c -c"
GCC_ONLY="-fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-finite-math-only -fno-unsafe-math-optimizations -fmath-errno -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -Wall -Wno-unknown-pragmas -pedantic"

echo; echo "== g++ with a finite 1 GiB stack limit"
(ulimit -s 1048576; echo "ulimit -s: $(ulimit -s)"; /usr/bin/time -f "rc=%x wall=%es maxrss=%MkB" g++ "$W" $COMMON $GCC_ONLY -o /tmp/w_gcc.o 2>&1 | grep -E "rc=|internal compiler error| error:" | head -5)
echo; echo "== clang++ $(clang++ --version | head -1)"
/usr/bin/time -f "rc=%x wall=%es maxrss=%MkB" clang++ "$W" $COMMON -fno-trapping-math -ffp-contract=off -Wno-unknown-pragmas -o /tmp/w_clang.o 2>&1 | grep -E "rc=|error|Stack dump|PLEASE" | head -8
ls -la /tmp/w_gcc.o /tmp/w_clang.o 2>&1
