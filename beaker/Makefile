, := ,

CLUSTER   ?=
WORKSPACE ?= ai2/general-tool-use
BUDGET    ?= ai2/oe-omai
IMAGE     ?=
GPUS      ?=
CODE      ?=
SESSION   ?=

.PHONY: interactive params list_secrets rl

params:
	@echo "CLUSTER   = $(CLUSTER)"
	@echo "WORKSPACE = $(WORKSPACE)"
	@echo "BUDGET    = $(BUDGET)"
	@echo "IMAGE     = $(IMAGE)"
	@echo "GPUS      = $(GPUS)"
	@echo "CODE      = $(CODE)"
	@echo "SESSION   = $(SESSION)"

rl:
	uv run rl @ configs/gsm8k_with_math_python_tool/rl.toml

list_secrets:
	beaker secret list --workspace $(WORKSPACE)

interactive: params
	beaker session create \
	--bare \
	--workspace $(WORKSPACE) \
	--budget $(BUDGET) \
	--env UV_PROJECT_ENVIRONMENT=/tmp/venv \
	$(if $(CLUSTER),--remote --cluster $(CLUSTER),) \
	$(if $(IMAGE),--image $(IMAGE),) \
	$(if $(GPUS),--gpus $(GPUS),) \
	--mount src=weka,ref=oe-training-default,dst=/weka \
	--mount src=secret,ref=tanmayg_ssh_key,dst=/root/.ssh/ssh_key \
	$(if $(CODE),--mount src=hostpath$(,)ref=$(CODE)$(,)dst=/root/$(notdir $(CODE)),)
