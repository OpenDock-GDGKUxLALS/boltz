# Load and run the AOTInductor package under gdb on a GPU container; print the instruction that traps.
# modal run cuda_modal.py::debug --script scratch/sigill_gdb.sh --gpu L40S
exec > >(tee /root/.boltz/artifacts/sigill_gdb.txt) 2>&1
echo "host flags: $(grep -m1 -o -wE 'avx512[a-z_]*|amx[a-z_]*|avx_vnni|avx2' /proc/cpuinfo | sort -u | tr '\n' ' ')"
nvidia-smi --query-gpu=name --format=csv,noheader
cat > /tmp/run.py <<'PY'
import torch
x = {k: v.cuda() for k, v in torch.load("/root/.boltz/artifacts/batch_prot.pt").items()}
run = torch._inductor.aoti_load_package("/root/.boltz/artifacts/prot_aoti.pt2")
print("loaded", flush=True)
torch.manual_seed(0); out = run(x); torch.cuda.synchronize()
print("ran: ptm", out["ptm"].tolist(), flush=True)
PY
gdb -q -batch -ex "handle SIGSEGV nostop noprint pass" -ex run -ex "info symbol \$pc" -ex "x/3i \$pc" -ex "bt 8" --args python3 /tmp/run.py 2>&1 \
  | grep -vE "^\[(New|Thread|Detaching)|Inferior|libthread|warning: (Unable|Error)" | tail -25
