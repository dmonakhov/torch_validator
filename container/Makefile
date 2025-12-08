IMAGE_NAME ?= torch-validator
IMAGE_TAG ?= latest
STEPS ?= 3000

.PHONY: build smoke-test test_local_determinism clean

build:
	docker build --progress plain --rm -t $(IMAGE_NAME):$(IMAGE_TAG) -f Dockerfile ..

# Meta-target: runs all smoke tests
smoke-test: test_local_determinism

test_local_determinism: build
	@echo "Running local determinism test (STEPS=$(STEPS), auto-detect GPUs)..."
	docker run --rm --gpus all \
		-e STEPS=$(STEPS) \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		bash -c 'NPROC=$$(nvidia-smi -L | wc -l) ./scripts/test_local_determinism.sh'

clean:
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
