\name{kminit}
\alias{kminit}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Set initial value for EM algorithm.
}
\description{
Set initial value for EM algorithm using k-means.
}
\usage{
kminit(y,L,...,seed = sample.int(.Machine$integer.max, 1),cluster=FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{y}{numeric matrix of data}
  \item{L}{number of clusters}
  \item{seed}{seed of random number generator. A single value, interpreted as an integer.}
  \item{cluster}{If cluster = FALSE (default), the function not return cluster indicator. If cluster = TRUE, the function return cluster indicator.}
  \item{...}{other options for 'kmeans' function}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
  \item{mu}{matrix of cluster centres}
  \item{var}{covariance matrices}
  \item{cluster}{indicator variables of clusters}
  \item{kmres}{return of 'kmeans' function}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
Ko ABE
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
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
% \keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
% \keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
