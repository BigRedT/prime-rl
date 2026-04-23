WORKSPACE ?= ai2/general-tool-use
BUDGET    ?= ai2/oe-omai
CLUSTER   ?=
IMAGE     ?=
GPUS      ?=

.PHONY: interactive params list_secrets rl rl_clean

params:
	@echo "WORKSPACE = $(WORKSPACE)"
	@echo "BUDGET    = $(BUDGET)"
	@echo "CLUSTER   = $(CLUSTER)"
	@echo "IMAGE     = $(IMAGE)"
	@echo "GPUS      = $(GPUS)"

rl:
	uv run rl @ configs/gsm8k_with_math_python_tool/rl.toml

rl_clean:
	uv run rl @ configs/gsm8k_with_math_python_tool/rl.toml --clean-output-dir

list_secrets:
	beaker secret list --workspace $(WORKSPACE)

interactive: params
	beaker session create \
	--bare \
	--host-networking \
	--workspace $(WORKSPACE) \
	--budget $(BUDGET) \
	--env UV_PROJECT_ENVIRONMENT=/tmp/venv \
	--mount src=weka,ref=oe-training-default,dst=/weka \
	--mount src=secret,ref=tanmayg_ssh_key,dst=/root/.ssh/ssh_key \
	$(if $(CLUSTER),--remote --cluster $(CLUSTER),) \
	$(if $(IMAGE),--image $(IMAGE),) \
	$(if $(GPUS),--gpus $(GPUS),)
