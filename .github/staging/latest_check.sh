#!/usr/bin/env bash
# Round 7: check the published Studio image (unsloth/unsloth:latest) on a host
# without a GPU. Asserts the image revision, the CUDA llama.cpp bundle kept by
# #10495 (marker + both libggml-cuda.so paths), real CPU inference through that
# bundle (llama-bench + llama-completion on a 0.6B Q4 GGUF), the Python imports,
# and optionally the Studio + JupyterLab boot.
#
# usage: latest_check.sh IMAGE MACHINE REVISION [STUDIO=0|1]
set -eu
IMAGE=$1; MACHINE=$2; REV=$3; STUDIO=${4:-0}
GGUF_URL=https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf

fail() { echo "::error::$*"; exit 1; }

echo "== pull"
docker pull "$IMAGE" 2>&1 | tail -2
INS=$(docker image inspect "$IMAGE" --format 'arch={{.Architecture}} size={{.Size}} revision={{index .Config.Labels "org.opencontainers.image.revision"}}')
echo "$INS"
echo "$INS" | grep -q "revision=$REV" || fail "image revision is not $REV"
df -h / | tail -1 || true

echo "== llama.cpp bundle"
OUT=$(docker run -i --rm --entrypoint bash "$IMAGE" -c '
  set -e
  python3 -c "import json;d=json.load(open(\"/opt/unsloth/llama.cpp/UNSLOTH_PREBUILT_INFO.json\"));print(\"marker\",d.get(\"platform\"),d.get(\"bundle_profile\"),d.get(\"backend\"),d.get(\"tag\"))"
  for f in /opt/unsloth/llama.cpp/libggml-cuda.so /opt/unsloth/llama.cpp/build/bin/libggml-cuda.so; do
    if [ -f "$f" ]; then echo "present $f"; else echo "MISSING $f"; fi
  done
  ls /opt/unsloth/llama.cpp/*.so | xargs -n1 basename | grep -c "^libggml-cpu-" | sed "s/^/cpu backends: /"
  /opt/unsloth/llama.cpp/build/bin/llama-server --version 2>&1 | tail -1
  /opt/unsloth/llama.cpp/build/bin/llama-server --list-devices 2>&1 | tail -3')
echo "$OUT"
echo "$OUT" | grep -q "^marker linux-.*-cuda cuda12-portable" || fail "marker is not the CUDA portable bundle (the CPU prebuilt replaced it)"
[ "$(echo "$OUT" | grep -c '^present ')" = 2 ] || fail "libggml-cuda.so missing from the image"

echo "== CPU inference through the CUDA bundle"
OUT=$(docker run -i --rm --entrypoint bash "$IMAGE" -c '
  set -e
  python3 - <<PY
import urllib.request, sys
for attempt in range(3):
    try:
        urllib.request.urlretrieve("'"$GGUF_URL"'", "/tmp/m.gguf"); break
    except Exception as e:
        print("download retry", attempt, e, file=sys.stderr)
else:
    sys.exit("gguf download failed")
PY
  ls -la /tmp/m.gguf
  B=/opt/unsloth/llama.cpp/build/bin
  $B/llama-bench -m /tmp/m.gguf -p 32 -n 16 -t 2 -o md 2>&1 | grep -E "load_backend|CPU|build:"
  $B/llama-completion -m /tmp/m.gguf -p "The capital of France is" -n 12 -t 2 --temp 0 -no-cnv 2>&1 | grep -iE "load_backend|capital of France"')
echo "$OUT"
echo "$OUT" | grep -q "load_backend: loaded CPU backend" || fail "CPU backend did not load"
echo "$OUT" | grep -qi "loaded CUDA backend" && fail "CUDA backend loaded on a host without a GPU"
echo "$OUT" | grep -q "| CPU " || fail "llama-bench did not run on the CPU backend"
echo "$OUT" | grep -q "Paris" || fail "completion did not produce Paris"

echo "== imports (CPU mode)"
OUT=$(docker run -i --rm -e UNSLOTH_ALLOW_CPU=1 -e UNSLOTH_SKIP_NOTEBOOK_SYNC=1 "$IMAGE" python - <<'PY'
import platform, torch, unsloth, subprocess
print("machine", platform.machine(), "torch", torch.__version__, "cuda", torch.cuda.is_available(), "unsloth", unsloth.__version__)
print("cli:", subprocess.run(["unsloth", "--version"], capture_output=True, text=True).stdout.strip())
PY
)
echo "$OUT"
echo "$OUT" | grep -q "machine $MACHINE torch 2.11" || fail "import output missing or wrong arch"

if [ "$STUDIO" = 1 ]; then
  echo "== Studio + JupyterLab boot"
  CID=$(docker run -d -e UNSLOTH_ALLOW_CPU=1 -e UNSLOTH_SKIP_NOTEBOOK_SYNC=1 \
        -e UNSLOTH_STUDIO_PASSWORD=ci-round7-password -e JUPYTER_PASSWORD=ci-round7-jupyter \
        -p 127.0.0.1:8000:8000 -p 127.0.0.1:8888:8888 "$IMAGE")
  for i in $(seq 1 240); do
    docker logs "$CID" 2>&1 | grep -q "Unsloth container ready" && break
    if ! docker ps -q --no-trunc | grep -q "$CID"; then echo "::error::container exited"; docker logs "$CID" 2>&1 | tail -60; exit 1; fi
    sleep 5
  done
  docker logs "$CID" 2>&1 | grep -E "Unsloth Studio login|Unsloth container|Studio  |JupyterLab|llama" | head -10
  docker logs "$CID" 2>&1 | grep -q "Unsloth container ready" || { echo "::error::ready block never appeared"; docker logs "$CID" 2>&1 | tail -80; exit 1; }
  HEALTH=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/api/health)
  LOGIN=$(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:8000/api/auth/login -H 'Content-Type: application/json' -d '{"username":"unsloth","password":"ci-round7-password"}')
  BADLOGIN=$(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:8000/api/auth/login -H 'Content-Type: application/json' -d '{"username":"unsloth","password":"wrong"}')
  JUP=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8888/login)
  echo "studio health: $HEALTH  login: $LOGIN  wrong password: $BADLOGIN  jupyter login page: $JUP"
  docker rm -f "$CID" >/dev/null
  test "$HEALTH" = 200 && test "$LOGIN" = 200 && test "$BADLOGIN" != 200 && test "$JUP" = 200 || fail "Studio boot checks failed"
fi
echo "== all checks passed on $MACHINE"
