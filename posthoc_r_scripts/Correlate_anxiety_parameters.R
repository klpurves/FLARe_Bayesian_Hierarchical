

## correlate the paramers extracted with eachotehr, anxiety and learning / IQ metrics.

## read in the anxiety / learning data 

library(data.table)
library(psych)
library(ggcorrplot)

anx <- fread("/Users/kirstin/Dropbox/SGDP/FLARe/FLARe_MASTER/Projects/LatentGrowth/Datasets/acq_cases_anxiety_IQ.csv",data.table=F)

## when final model is decided and saved then read in here. for now do this when the script has run and I have a column per parameter

dat <- cbind(anxBac,alpha_est, beta, lambda)

## make all levels numeric


  
for (col in 2:17){
  
  dat[[col]] <- as.numeric(levels(dat[[col]]))[dat[[col]]] 
  
}
  
## correlation matrix of all

mat <- round(cor(dat[2:dim(dat)[2]]),1)

p.mat <- cor_pmat(dat[2:dim(dat)[2]])


# Reordering the correlation matrix
# --------------------------------
# using hierarchical clustering

ggcorrplot(mat, 
           p.mat = p.mat,
          # insig = "blank",
           lab = TRUE,
           hc.order = T,
           outline.col = "white",
           method='circle',
           type = "upper",
           ggtheme = ggplot2::theme_gray,
           colors = c("#6D9EC1", "white", "#E46726"),
          show.diag = FALSE)


