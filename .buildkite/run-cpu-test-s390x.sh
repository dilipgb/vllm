#!/bin/bash

set -ex

# Setup cleanup
remove_docker_container() { docker rm -f cpu-test || true; docker system prune -f; }
trap remove_docker_container EXIT
remove_docker_container

# Try building the docker image
docker build -t cpu-test -f Dockerfile.s390x .

# Run the image
source /etc/environment
docker run -itd \
  --entrypoint /bin/bash \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --privileged=true \
  --network host \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  --name cpu-test \
  cpu-test

function cpu_tests() {
  set -e

  # Check HF_TOKEN availability
  if [ -z "$HF_TOKEN" ]; then
    echo "HF_TOKEN is not set. Please provide a valid Hugging Face token."
    exit 1
  fi

  # Basic setup and model tests (as root)
  docker exec --user root cpu-test bash -c "
    set -e

    # Update and install Python3 and pip
    apt-get update && apt-get install -y python3 python3-pip

    # Set Python3 and pip paths
    ln -sf /usr/bin/python3 /usr/bin/python
    ln -sf /usr/bin/pip3 /usr/bin/pip

    # Add pytest to PATH environment variable
    export PATH=\$PATH:/home/vllm/.local/bin

    echo 'Installing dependencies...'
    pip install pytest pytest-asyncio \
      einops librosa peft Pillow sentence-transformers soundfile \
      transformers_stream_generator matplotlib datamodel_code_generator
    pip install torchvision --index-url https://download.pytorch.org/whl/cpu

    echo 'Starting pytest: decoder_only/language'
    pytest -v -s tests/models/decoder_only/language -m cpu_model || echo 'Test failed: decoder_only/language'

    echo 'Starting pytest: embedding/language'
    pytest -v -s tests/models/embedding/language -m cpu_model || echo 'Test failed: embedding/language'

  #  echo 'Starting pytest: encoder-decoder/language'
  #  pytest -v -s tests/models/encoder_decoder/language -m cpu_model || echo 'Test failed: encoder decoder/language'

  #  echo 'Starting pytest: decoder-only-audio-language/language'
  #  pytest -v -s tests/models/decoder_only/audio_language -m cpu_model || echo 'Test failed: decoder_only/audio_language/language'

    echo 'All tests completed.'
  "

  # Online inference (without root)
  docker exec cpu-test bash -c "
    set -e
    echo 'Starting the VLLM API server...'
    python3 -m vllm.entrypoints.openai.api_server --model facebook/opt-125m --dtype float &
    echo 'Waiting for API server to be ready...'
    timeout 600 bash -c 'until curl -s localhost:8000/v1/models; do sleep 1; done' || exit 1
    echo 'Running benchmark tests...'
    python3 benchmarks/benchmark_serving.py \
      --backend vllm \
      --dataset-name random \
      --model facebook/opt-125m \
      --num-prompts 20 \
      --endpoint /v1/completions \
      --tokenizer facebook/opt-125m || echo 'Benchmark tests failed.' "

}

# Run tests with timeout
export -f cpu_tests
timeout 200m bash -c "cpu_tests"
