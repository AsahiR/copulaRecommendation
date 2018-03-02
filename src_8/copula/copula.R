library('copula')

#This function returns copula function
select.copula <- function(cop.name, dimention) {
    if(cop.name == 'frank'){
        function(theta) {
            frankCopula(theta, dim=dimention)
        }
    } else if(cop.name == 'clayton'){
        function(theta) {
            claytonCopula(theta, dim=dimention)
        }
    } else if(cop.name == 'gumbel'){
        function(theta) {
            gumbelCopula(theta, dim=dimention)
        }
    } else if(cop.name == 'normal'){
        function(theta) {
            normalCopula(theta, dim=dimention)
        }
    }
}

#This funtion optimizes the parameter of copula and return it.
# Type of copula (cop.name) and dimention of copula (dimention) are required because range of the parameter is dicided by the type and dimention.
optimize.param <- function(log.likelihood.func, cop.name, dimention) {
    if(cop.name == 'frank'){
        if (dimention == 2) {
            param.upper <- 1000
            param.lower <- -1000
        } else {
            param.upper <- 1000
            param.lower <- 0
        }
    } else if(cop.name == 'clayton'){
        if (dimention == 2) {
            param.upper <- 1000
            param.lower <- -1
        } else {
            param.upper <- 1000
            param.lower <- 0
        }
    } else if(cop.name == 'gumbel'){
        param.upper <- 1000
        param.lower <- 1
    } else if(cop.name == 'normal'){
        param.upper <- 1
        param.lower <- -1
    }
    result <- optimize(f=log.likelihood.func, c(param.lower,param.upper), maximum=TRUE)
    return(result$maximum)
}

#This constructs optimized copula from training data. You have to choice the type of copula(frank, clayton, gumbel or normal).
copula.constructor <- function(training.marginal.dist.matrix, cop.name, param, dim) {
    dimention <- ncol(training.marginal.dist.matrix)
    if (!is.null(dim)) {
        dimention <- dim
    }
    selected.copula <- select.copula(cop.name, dimention)
    if (!is.null(param)) {
        optimized.cop <- selected.copula(param)
        return (optimized.cop)
    }
    log.density.func <- function(cop) {
        log(pCopula(training.marginal.dist.matrix, cop))
    }
    #Create log likelihood func for optimizing parameter
    log.likelihood.func <- function(theta) {
        cop <- selected.copula(theta)
        log.density <- log.density.func(cop)
        L <- 1 + sum(sapply(log.density,
        function(d) { if (is.na(d)) return (0) else if (d == Inf) return (1000) else if (d == -Inf) return (-1000) else return (d)}))
        return (L)
    }
    optimized.param <- optimize.param(log.likelihood.func, cop.name, dimention)
    optimized.cop <- selected.copula(optimized.param)
}

if (py.cop.name == 'indep') {
    trained <- indepCopula(dim=ncol(training.marginal.dist.matrix))
    trained.param <- 'indep'
} else {
    trained <- copula.constructor(py.training.marginal.dist.matrix, py.cop.name, py.param, py.dim)
    trained.param <- trained@parameters[1]
}
