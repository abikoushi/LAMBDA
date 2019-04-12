\name{mixtruncGaussEM}
\alias{mixtruncGaussEM}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Model based clustering for mass cytometry data.
}
\description{
Parameter estimation for zero-inflated Gaussian mixture model.
}
\usage{
mixtruncGaussEM(Y, X, L, muini, betaini, Sigmaini, tau, nu, Lambda, num_iter)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{Y}{mass cytometry data}
  \item{X}{covariate matrix}
  \item{L}{number of clusters}
  \item{muini}{initial value of mean parameters}
  \item{betaini}{inital value of coefficients parameters}
  \item{Sigmaini}{initial value of variance parameters}
  \item{tau}{precision parameter of Gauss-Wishart prior distribution}
  \item{nu}{degree of freedom parameter of Wishart prior distribution}
  \item{Lambda}{scale parameter of Wishart prior distribution}
  \item{num_iter}{number of iteration}
}
%%\details{
%%  ~~ If necessary, more details than the description above ~~
%%}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
\item{beta}{coefficients}
\item{mu}{mean}
\item{Sigma}{covariance matrices}
\item{loglik}{log likelihood which evaluated by each iterations}
\item{W}{indicator variables of clusters}
%% ...
}
\references{
Model based clustering for flow and mass cytometry data with clinical information
}
\author{
Ko ABE
}
%%\note{
%%  ~~further notes~~
%%}
%% ~Make other sections like Warning with \section{Warning }{....} ~
%%\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
%%}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function (x) 
{
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
%%\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
%%\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
