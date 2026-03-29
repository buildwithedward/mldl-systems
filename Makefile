.PHONY: new

DAY ?=

new:
ifndef DAY
	$(error Usage: make new DAY=day-1)
endif
	@if [ ! -f $(DAY)/filetree.md ]; then \
		echo "Error: $(DAY)/filetree.md not found. Create it first with the desired folder structure."; \
		exit 1; \
	fi
	@sed 's/├/+/g; s/└/`/g; s/──/-/g; s/│/|/g' $(DAY)/filetree.md \
		| awk '{ \
			line = $$0; \
			indent = 0; \
			tmp = line; \
			while (match(tmp, /^(\|   |\+- |`- |    )/)) { \
				indent++; \
				tmp = substr(tmp, RLENGTH+1); \
			} \
			gsub(/^[|+` -]+/, "", tmp); \
			gsub(/^[ \t]+/, "", tmp); \
			gsub(/[ \t]+$$/, "", tmp); \
			if (tmp == "") next; \
			stack[indent] = tmp; \
			path = ""; \
			for (i=1; i<=indent; i++) { path = path stack[i]; } \
			if (tmp ~ /\/$$/) { \
				if (path != "") print "d:" path; \
			} else { \
				if (path == "") path = tmp; \
				print "f:" path; \
			} \
		}' | while IFS= read -r entry; do \
			type=$${entry%%:*}; \
			p=$${entry#*:}; \
			if [ "$$type" = "d" ]; then \
				mkdir -p "$(DAY)/$$p"; \
			elif [ "$$type" = "f" ]; then \
				dir=$$(dirname "$$p"); \
				[ "$$dir" != "." ] && mkdir -p "$(DAY)/$$dir"; \
				touch "$(DAY)/$$p"; \
			fi; \
		done
	@echo "Created $(DAY)/ structure from filetree.md"
