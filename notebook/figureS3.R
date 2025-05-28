d<-read.delim("2025_05_12_protein_quantitation_table_sparks_SP.txt", header=T, sep="\t")
d<-d[rev(order(d$IBAQ_SEC_fraction_56)),]
danwoldo<-subset(d, d$ProteinAccessions=="J2ZJU5")

d<-subset(d, d$IBAQ_norm_SEC_fraction_56==1 | d$IBAQ_norm_SEC_fraction_57==1)
d<-subset(d, d$signal_peptide!="")
#top one is histone... 
plot(1,1, type="n", xlim=c(51.5, 65.5), ylim=c(0, max(d$IBAQ_SEC_fraction_57)+100000))
for (i in 1:length(d[[1]])) {
points(52:65, d[i,20:33], typ="l")
}

pdf("danwoldo_peaks.pdf")


plot(1,1, type="n", xlim=c(51.5, 65.5), ylim=c(0, max(danwoldo$IBAQ_SEC_fraction_56)+100), xlab="Fractions", ylab="Protein abundance (IBAQ)", main="Danwoldo")
abline(v=52:65, col="grey", lty=3)
points(52:65, danwoldo[1,20:33], typ="l", col="black", lwd=1.5)
dev.off()