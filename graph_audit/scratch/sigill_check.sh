# Why the AOTInductor package dies with SIGILL on another container (REPORT section 7).
# modal run cuda_modal.py::debug --script scratch/sigill_check.sh
exec > >(tee /root/.boltz/artifacts/sigill_check.txt) 2>&1
mkdir -p /tmp/pkg && cd /tmp/pkg && python3 -c "import zipfile; zipfile.ZipFile('/root/.boltz/artifacts/prot_aoti.pt2').extractall('.')"
echo "this host: $(grep -m1 'model name' /proc/cpuinfo); avx512f: $(grep -m1 -o -w avx512f /proc/cpuinfo || echo no)"
for so in $(find /tmp/pkg -name '*.so'); do
  echo "$so: $(objdump -d --no-show-raw-insn "$so" | grep -cE '%zmm|\{%k[0-7]\}') AVX-512 instructions"
  objdump -d --no-show-raw-insn "$so" | grep -m3 -E '%zmm' | sed 's/^/    /'
done
