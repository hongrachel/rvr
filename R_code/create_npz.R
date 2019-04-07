# Create esets via curatedovariandata package

library(curatedOvarianData)
# From curatedOvarianData vignette
source(system.file("extdata", "patientselection.config",package="curatedOvarianData"))
sapply(ls(), function(x) if(!x %in% c("remove.samples", "duplicates")) print(get(x)))
source(system.file("extdata", "createEsetList.R", package = "curatedOvarianData"))

# Save the original eset list - reuse this list for other analyses
# e.g. save(esets, file = "esets.Rda")

library(reticulate)

library(simulatorZ)

# An event in this setting is death before the cutoff

surv_to_bin <- function(surv, cutoff = 730){

	bin <- vector("numeric", nrow(surv))
	for(i in 1:nrow(surv)){
		if(surv[i,2] == 0){
			if(surv[i,1] >= cutoff){
				bin[i] <- 0
			} else {
				bin[i] <- 1
			}
		} else {
			bin[i] <- 1
		}
	}

	cbind(bin, abs(bin - 1))
}

# .npz components from laftr example files
# x_train: n1 x p matrix of training x attributes
# x_test: n2 x p matrix of test set x attributes
# y_train: n1 x 2 matrix of 0/1 outcomes
# y_test: n2 x 2 matrix of 0/1 outcomes
# attr_train: n1 x 1 matrix of sensitive attribute status
# attr_test: n2 x 1 matrix of sensitive attribute status
# train_inds: 80% of IDs from train internal training
# valid_inds: 20% of IDs from train for internal validation

# Training dataset is TCGA + GSE9891
# Test dataset is GSE26193 + GSE26712

rowidx <- Reduce(intersect, list(rownames(esets$TCGA_eset), rownames(esets$GSE9891),
	 		  rownames(esets$GSE26193), rownames(esets$GSE26712)))

xtrain_1 <- exprs(esets$TCGA_eset)[rowidx,]
xtrain_2 <- exprs(esets$GSE9891)[rowidx,]
xtest_1 <- exprs(esets$GSE26193)[rowidx,]
xtest_2 <- exprs(esets$GSE26712)[rowidx,]

ytrain_1 <- surv_to_bin(pData(esets$TCGA_eset)$y)
ytrain_2 <- surv_to_bin(pData(esets$GSE9891)$y)
ytest_1 <- surv_to_bin(pData(esets$GSE26193)$y)
ytest_2 <- surv_to_bin(pData(esets$GSE26712)$y)

tmp <- rowCoxTests(xtrain_1, pData(esets$TCGA_eset)$y)
coxrows <- rownames(tmp)[order(abs(tmp[,1]), decreasing = T)[1:100]]

xtrain_1 <- xtrain_1[coxrows,]
xtrain_2 <- xtrain_2[coxrows,]
xtest_1 <- xtest_1[coxrows,]
xtest_2 <- xtest_2[coxrows,]

x_train <- rbind(t(xtrain_1), t(xtrain_2))
x_test <- rbind(t(xtest_1), t(xtest_2))
y_train <- rbind(ytrain_1, ytrain_2)
y_test <- rbind(ytest_1, ytest_2)
attr_train <- as.matrix(c(rep(0, ncol(xtrain_1)), rep(1, ncol(xtrain_2))))
attr_test <- as.matrix(c(rep(0, ncol(xtest_1)), rep(1, ncol(xtest_2))))
pct_idx <- round(0.8*nrow(x_train))
inds <- sample(0:(nrow(x_train)-1))
train_inds <- inds[1:pct_idx]
test_inds <- inds[(pct_idx+1):nrow(x_train)]

outfile = "new_esets.npz"

np <- import("numpy")
np$savez(outfile, x_train = x_train, x_test = x_test, y_train = y_train,
			y_test = y_test, attr_train = attr_train, attr_test = attr_test,
			train_inds = train_inds, valid_inds = test_inds)