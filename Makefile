.PHONY: new

DAY ?=

new:
ifndef DAY
	$(error Usage: make new DAY=day-1)
endif
	@mkdir -p $(DAY)/src $(DAY)/tests $(DAY)/data $(DAY)/models
	@touch $(DAY)/.env
	@touch $(DAY)/.env.example
	@touch $(DAY)/.gitignore
	@touch $(DAY)/Dockerfile
	@touch $(DAY)/Makefile
	@touch $(DAY)/requirements.txt
	@touch $(DAY)/src/__init__.py $(DAY)/tests/__init__.py $(DAY)/src/config.py $(DAY)/src/exceptions.py
	@echo "Created $(DAY)/ with default structure"
