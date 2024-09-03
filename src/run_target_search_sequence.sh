phmmer \
	--noali \
	-o /dev/null \
	--domtblout ../data/putative_targets.domtblout.txt \
	--cpu 32 \
	--domE 1e-6 \
	../data/pgh_domains_archaea.fasta \
	../data/pgh_db.fasta
