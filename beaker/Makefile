, := ,

CLUSTER   ?=
WORKSPACE ?= ai2/tanmayg
BUDGET    ?= ai2/oe-base
IMAGE     ?=
GPUS      ?=
CODE      ?=
SESSION   ?=

.PHONY: interactive params

params:
	@echo "CLUSTER   = $(CLUSTER)"
	@echo "WORKSPACE = $(WORKSPACE)"
	@echo "BUDGET    = $(BUDGET)"
	@echo "IMAGE     = $(IMAGE)"
	@echo "GPUS      = $(GPUS)"
	@echo "CODE      = $(CODE)"
	@echo "SESSION   = $(SESSION)"

interactive: params
	beaker session create \
	--bare \
	--workspace $(WORKSPACE) \
	--budget $(BUDGET) \
	$(if $(CLUSTER),--remote --cluster $(CLUSTER),) \
	$(if $(IMAGE),--image $(IMAGE),) \
	$(if $(GPUS),--gpus $(GPUS),) \
	--mount src=weka,ref=oe-training-default,dst=/weka \
	$(if $(CODE),--mount src=hostpath$(,)ref=$(CODE)$(,)dst=/root/$(notdir $(CODE)),) \
	--secret-env SSH_KEY=SSH_KEY
