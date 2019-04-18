# K: number of studies
# nk: sample size of each study
# p: number of covariates
# p_c: number of common covariates (c <= p)
# mu - K*p matrix of covariate means
# SIG - p*p covariance matrix
# eps - perturbation level for homogenous covariates
# eta - perturbation level for heterogeneous covariates
# beta_min - minimum for beta window
# beta_max - maximum for beta window

# Beta are generated from [-beta_max, -beta_min] U [beta_min, beta_max]
# for beta_k in p_c, the study-specific beta_k is taken from [beta_k - eps, beta_k + eps]
# for beta_k not in p_c, the study-spceific beta_k is taken from [beta_k - eta, beta_k + eta]

MultiStudySim <- function(K, nk, p, p_c, mu, SIGMA, eps, eta, beta_min, beta_max){
	# Generate X matrices
	x_vec_list <- lapply(1:K,function(x) mvrnorm(n = nk[x], mu[x,], SIG))

	# indices of 'common features'
	c_idx <- sample(1:p, p_c)

	# Generate 'true' betas
	beta_vec <- sample(c(runif(round(p/2), -beta_max, -beta_min), runif(p - round(p/2), beta_min, beta_max)))

	# Generate study-specific betas
	beta_vec_list <- lapply(1:K,function(x){
		
		bvtmp <- beta_vec
		bvtmp[c_idx] <- sapply(bvtmp[c_idx], function(z){runif(1, z - eps, z + eps)})
		bvtmp[-c_idx] <- sapply(bvtmp[-c_idx], function(z){runif(1, z - eta, z + eta)})
		bvtmp
	})

	# Outcome model with logit link
	output_Y_list <- lapply(1:K,function(x){
		xb <- x_vec_list[[x]]%*%beta_vec_list[[x]]
		z <- 1/(1 + exp(-xb))
		rbinom(length(z), 1, z)	
	})

	# List of final data and beta values should we want to check them
	final_data <- lapply(1:K,function(x) list(SimulatedOutput=data.frame(x_vec_list[[x]],y=output_Y_list[[x]],row.names = c(1:length(output_Y_list[[x]]))),
				   BetaValue=beta_vec_list[[x]]))
  
	names(final_data) <- paste0('Study',c(1:K))
	return(final_data)
}

library(reticulate)
library(MASS)

# Set parameters for run
N <- 5 # Total number of studies
K <- 4 # number of training studies
nk <- rep(5000,N) # number of observations per study, currently all same
p <- 30 # number of covariates
p_c <- 20 # number of common covariates
eps <- 0.1 # window size for common covariates
eta <- 2 # window size for non-comman covariates
beta_min <- 1 # beta window minimum
beta_max <- 5 # beta window maximum

# covariate means
mu <- matrix(runif(N*p,-3,3),nrow=N,ncol=p,byrow=T)

# SIG diagonal
#SIG <- diag(x=1,nrow = p,ncol = p)

# SIG arbitrary
A <- matrix(runif(p^2)*2-1, ncol=p) 
SIG <- t(A) %*% A

# Generate training + testing sets
alldata <- MultiStudySim(N, nk, p, p_c, mu, SIGMA, eps, eta, beta_min, beta_max)

# Separate into training and testing, taking first N - K studies as test (arbitrary)
testing <- alldata[N - K]
training <- alldata[2:N]

# Format training data for .npz
x_train <- do.call(rbind, lapply(training, function(x){x[[1]][,1:p]}))
y_train <- do.call(c, lapply(training, function(x){x[[1]]$y}))
y_train <- cbind(y_train, 1 - y_train)

inds <- rep(1:K, each = nk[1])

attr_train <- model.matrix(~factor(inds)-1)

pct_idx <- round(0.8*nrow(x_train))
inds <- sample(0:(nrow(x_train)-1))
train_inds <- inds[1:pct_idx]
test_inds <- inds[(pct_idx+1):nrow(x_train)]

# Generate test set
# NOT RUN ANYMORE #
# testing <- MultiStudySim(1, nk, p, p_c, mu, SIGMA, eps, eta, beta_min, beta_max)

# Format testing data for .npz
x_test <- testing$Study1[[1]][,1:p]
y_test <- testing$Study1[[1]]$y
y_test <- cbind(y_test, 1 - y_test)

attr_test <- matrix(0, nrow(x_test), K)
attr_test[,1] <- rep(1, nrow(x_test))

x_train <- as.matrix(x_train)
x_test <- as.matrix(x_test)

outfile <- "run_p1_2_041719.npz"
np <- import("numpy")
np$savez(outfile, x_train = as.matrix(x_train), x_test = as.matrix(x_test), y_train = y_train,
			y_test = y_test, attr_train = attr_train, attr_test = attr_test,
			train_inds = train_inds, valid_inds = test_inds)

# Boxplot to check covariate ranges
boxplot(do.call(rbind, lapply(training, "[[", 2)), xlab = "beta", ylab = "coefficient value")
points(1:p, testing$Study1$BetaValue, col = "red", pch = 19, cex = 0.7)
