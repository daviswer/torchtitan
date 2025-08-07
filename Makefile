BATCH_SIZE ?= 2
STEPS ?= 1000
NGPU := $(shell nvidia-smi --list-gpus | wc -l)

define run_fsdp
	export NGPU=$(NGPU) && \
	export CONFIG_FILE=$(1) && \
	./run_llama_train.sh \
		--training.batch_size $(BATCH_SIZE) \
		--training.steps $(STEPS) \
		--training.seq_len $(2)
endef

help:
	@echo "Choose another target"

fsdp:
	$(call run_fsdp,./train_configs/debug_model.toml,2048)

fsdp-yarn:
	$(call run_fsdp,./train_configs/debug_yarn_model.toml,4096)
