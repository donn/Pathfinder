.PHONY: clean_logs
clean:
	rm -f ./logs/*

.PHONY: clean_presynth_yes_im_sure
clean_presynth_yes_im_sure:
	rm -f data/presynth_cache.yml
	rm -rf ./designs/*/runs

.PHONY: clean_pretrain_yes_im_sure
clean_presynth_yes_im_sure:
	rm -f data/pretrain.yml
	rm -rf ./designs/*/runs