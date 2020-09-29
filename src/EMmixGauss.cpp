// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
const double log2pi = std::log(2.0 * M_PI);

arma::rowvec rcate(const arma::rowvec & p){
  int K = p.n_cols;
  arma::rowvec cump = cumsum(p);
  arma::rowvec x(K);
  x.fill(0);
  double U = R::runif(0,1);
  if(U<=cump[0]){
    x[0] = 1;
  }else{
    for(int k=1; k<K; k++){
      if(cump[k-1]<U & U<=cump[k]){
        x[k] = 1;
      }
    }
  }
  return(x);
}

// sub1 returns a matrix x[-i,-i]
arma::mat sub1(arma::mat x, int i) {
  x.shed_col(i);
  x.shed_row(i);
  return x;
}

// sub2 returns a matrix x[a,-b]
arma::mat sub2(arma::mat x, int a, int b){
  x.shed_col(b);
  return(x.row(a));
}

// negSubCol returns a column vector x[-i]
arma::vec negSubCol(arma::vec x, int i){
  x.shed_row(i);
  return(x);
}

// negSubRow returns a row vector x[-i]
arma::rowvec negSubRow(arma::rowvec x, int i){
  x.shed_col(i);
  return(x);
}

arma::vec myrunif(int d){
  return arma::randu(d);
}

arma::vec rtmvnorm_gibbs(arma::vec mu, arma::mat omega, arma::vec init_state){
  // Rprintf("Start gibbs\n");
  int d = mu.n_elem; //check dimension of target distribution
  
  //draw from U(0,1)
  arma::vec U = arma::randu(d);
  
  //calculate conditional standard deviations
  double var;
  arma::vec x = init_state;
  for(int i=0; i<d; i++){
    if(init_state[i]<=0){
      var = 1/omega(i,i);
      double mu_i = mu(i) - var*arma::as_scalar(sub2(omega,i,i)*(negSubCol(x,i)-negSubCol(mu,i)));
      //transformation
      double Fb = R::pnorm5(0,mu_i,std::sqrt(var),true,false);
      x(i) = mu_i + std::sqrt(var) * R::qnorm5(U(i) * Fb + 1e-100,0.0,1.0,1,0); 
    }
  }
  return x;
}

double logsumexp(const arma::rowvec & x){
  double maxx = max(x);
  double out = maxx + std::log(sum(exp(x-maxx)));
  return out;
}

// [[Rcpp::export]]
arma::rowvec softmax(const arma::rowvec & x){
  double den = logsumexp(x);
  arma::rowvec res = x;
  if(arma::is_finite(den)){
    res = exp(res - den); 
  }else{
    res.fill(0);
    res.elem(arma::find(x==max(x))).fill(1);
    res = res/sum(res);
  }
  return res;
}

arma::vec triangl(const arma::mat& X){
  int n = X.n_cols;
  arma::vec res(n * (n-1) / 2);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      res(j + i * (i-1) / 2) = X(i, j);
    }
  }
  return res;
}


double mvnorm_lpdf(arma::rowvec x,
                    arma::vec mean,
                    arma::mat sigma) {
  int xdim = x.n_cols;
  arma::mat out;
  double rootdet = -0.5 * std::log(det(sigma));
  double constants = -0.5 * xdim * log2pi;
  arma::vec z = x.t() - mean;
  out  = constants - 0.5 * z.t()*arma::inv(sigma)*z + rootdet;
  return(out(0,0));
}

double mvnorm_lpdf_det(arma::rowvec x,
                   arma::vec mean,
                   arma::mat invsigma,
                   double rootdet){
  int xdim = x.n_cols;
  arma::mat out;
  double constants = -(static_cast<double>(xdim)/2.0) * log2pi;
  arma::vec A = x.t() - mean;
  out  = constants - 0.5 * A.t()*invsigma*A + rootdet;
  return(out(0,0));
}

double mvnorm_lpdf_inv_det(arma::vec x,
                       arma::vec mean,
                       arma::mat invsigma,
                       double rootdet){
  int xdim = x.n_rows;
  arma::mat out;
  double constants = -(static_cast<double>(xdim)/2.0) * log2pi;
  arma::vec A = x - mean;
  out  = constants - 0.5 * A.t()*invsigma*A + rootdet;
  return(arma::as_scalar(out));
}

arma::vec cov2corvec(arma::mat sigma){
  arma::vec sds = pow(arma::diagvec(sigma),-0.5);
  arma::vec corrmat = triangl(arma::diagmat(sds) * sigma * arma::diagmat(sds));
  return corrmat;
}

Rcpp::List simZ(const arma::mat & z_pre ,const arma::mat & mu,
               const arma::cube & invSigma, const int & L,
               const arma::mat & w){
  // Rprintf("Start ");
  int N = z_pre.n_rows;
  arma::mat z = z_pre;
  double ll = 0;
  arma::vec rootdet(L);
  for(int l=0; l<L; l++){
    rootdet(l) = std::log(det(invSigma.slice(l)))/2.0;
  }
  for(int n=0; n<N; n++){
    int ind = as_scalar(arma::find(w.row(n)==1));
    arma::vec tmpz = rtmvnorm_gibbs(mu.col(ind),invSigma.slice(ind),z_pre.row(n).t());
    z.row(n) = tmpz.t();
    ll += mvnorm_lpdf_inv_det(tmpz,mu.col(ind),invSigma.slice(ind),rootdet(ind));
  }
  return List::create(z,ll);
}

arma::mat simW(arma::mat Z, arma::mat X,
                  arma::mat beta0, arma::mat mu, arma::cube invsigma,int L){
  int N = Z.n_rows;
  int D = X.n_cols;
  arma::mat beta = beta0;
  beta.insert_cols(0,arma::zeros(D));
  arma::mat Xbeta = X * beta;
  arma::rowvec lp(L);
  arma::mat W(N,L);
  arma::vec rootdet(L);
  for(int l=0; l<L; l++){
    rootdet(l) = std::log(det(invsigma.slice(l)))/2.0;
  }
  double den = 0;
  for(int n=0; n<N; n++){
    den = logsumexp(Xbeta.row(n));
    for(int l=0; l<L; l++){
      lp(l) = Xbeta(n,l)-den+mvnorm_lpdf_inv_det(Z.row(n).t(),mu.col(l),invsigma.slice(l),rootdet(l));
    }
    W.row(n) = rcate(softmax(lp));
  }
  return W;
}

Rcpp::List Ecalc(arma::mat y, arma::mat X,
                arma::mat beta0, arma::mat mu, arma::cube sigma,
                int N, int L,int D){
  arma::mat beta = beta0;
  beta.insert_cols(0,arma::zeros(D));
  arma::mat Xbeta = X * beta;
  arma::rowvec lp(L);
  arma::mat Ez(N,L);
  double ll=0;
  arma::vec rootdet(L);
  arma::cube invsigma=sigma;
  for(int l=0; l<L; l++){
    rootdet(l) = -std::log(det(sigma.slice(l)))/2.0;
    invsigma.slice(l) = arma::inv(sigma.slice(l));
  }
  double den = 0;
  for(int n=0; n<N; n++){
    den = logsumexp(Xbeta.row(n));
    for(int l=0; l<L; l++){
      lp(l) = Xbeta(n,l)-den+mvnorm_lpdf_det(y.row(n),mu.col(l),invsigma.slice(l),rootdet(l)); 
    }
    Ez.row(n) = softmax(lp);
    ll += logsumexp(lp); 
  }
  return List::create(Rcpp::Named("Z")=Ez,_["loglik"]=ll);
}


double loglik(arma::mat y, arma::mat X,
                arma::mat beta0, arma::mat mu, arma::cube sigma,
                int N, int L,int D){
  arma::mat beta = beta0;
  beta.insert_cols(0,arma::zeros(D));
  arma::mat Xbeta = X * beta;
  arma::rowvec lp(L);
  arma::cube rooti = sigma; 
  double out=0;
  double den = 0;
  arma::vec rootdet(L);
  arma::cube invsigma=sigma;
  for(int l=0; l<L; l++){
    rootdet(l) = -std::log(det(sigma.slice(l)))/2.0;
    invsigma.slice(l) = arma::inv(sigma.slice(l));
  }
  for(int n=0; n<N; n++){
    den = std::log(sum(exp(Xbeta.row(n))));
    for(int l=0; l<L; l++){
      lp(l) = Xbeta(n,l)-den+mvnorm_lpdf_det(y.row(n),mu.col(l),invsigma.slice(l),rootdet(l)); 
    }
    out += logsumexp(lp); 
  }
  return out;
}

arma::vec weighted_colMeans(const arma::mat X, const arma::vec & w, const double & tau=0){
  int nCols = X.n_cols;
  arma::vec out(nCols);
  double den = sum(w);
  for(int i = 0; i < nCols; i++){
    out(i) = sum(w % X.col(i))/(den+tau);
  }
  return(out);
}


arma::cube sigma_updata(arma::mat y, arma::mat w, arma::mat mu, int L, double nu, arma::mat Lambda){
  int K = mu.n_rows;
  int N = y.n_rows;
  arma::cube Sigma(K,K,L);
  Sigma.fill(0);
  for(int l=0; l<L; l++){
    for(int n=0; n<N; n++){
      arma::vec d = y.row(n).t()-mu.col(l);
      Sigma.slice(l) += w(n,l)*d*d.t();
    }
  }
  for(int l=0; l<L; l++){
    Sigma.slice(l) = (Sigma.slice(l)+Lambda)/(sum(w.col(l)) + nu - K);
  }
  return Sigma;
}


arma::mat dQ1(arma::mat beta0,arma::mat X, arma::mat z, int D, int L, int N){
  arma::mat beta = beta0;
  beta.insert_cols(0,arma::zeros(D));
  arma::mat Xbeta = X*beta;
  arma::mat term2(D,L);
  arma::rowvec sXbeta(L);
  for(int n=0; n<N; n++){
    sXbeta = softmax(Xbeta.row(n));
  for(int l=0;l<L;l++){
   term2.col(l) += (X.row(n)*sXbeta(l)).t();
  }
  }
  arma::mat out = X.t()*z-term2;
  out.shed_col(0);
  return out;
}

Rcpp::List dQ1_d2Q1(arma::mat beta0, arma::mat X, arma::mat w, int D, int L, int N){
  arma::mat w2 = w;
  arma::mat beta = beta0;
  beta.insert_cols(0,arma::zeros(D));
  arma::mat Xbeta = X*beta;
  arma::mat grad(1,D*(L-1));
  arma::mat tmp;
  grad.fill(0);
  arma::mat hessian(D*(L-1),D*(L-1));
  hessian.fill(0);
  arma::rowvec sXbeta(L);
  w2.shed_col(0);
  for(int n=0; n<N; n++){
    sXbeta = softmax(Xbeta.row(n));
    sXbeta.shed_col(0);
    arma::mat Lp = diagmat(sXbeta);
    hessian += arma::kron(Lp-sXbeta.t()*sXbeta,X.row(n).t()*X.row(n));
    grad += arma::kron(w2.row(n)-sXbeta,X.row(n));
  }
  return List::create(-grad,hessian);
}

// [[Rcpp::export]]
Rcpp::List mixGaussEM(arma::mat Y, arma::mat X, int L, arma::mat muini, arma::mat betaini, arma::cube Sigmaini,
                      const double & tau, const double & nu, const arma::mat & Lambda,
                      int num_iter){
  arma::mat mu = muini;
  arma::mat beta = betaini;
  arma::cube Sigma = Sigmaini;
  int N = Y.n_rows;
  int D = X.n_cols;
  Rcpp::List L1;
  arma::mat dbeta(D,L);
  arma::vec ll_hist(num_iter);
  List beta_grad(2);
  for(int h=0;h<num_iter;h++){
    L1 = Ecalc(Y,X,beta,mu,Sigma,N,L,D);
    arma::mat Z = L1[0];
    beta_grad = dQ1_d2Q1(beta,X,Z,D,L,N);
    arma::mat dbeta = beta_grad[0];
    arma::mat d2beta = beta_grad[1];
    beta = arma::vectorise(beta,0) - arma::pinv(d2beta)*arma::vectorise(dbeta,0);
    beta.reshape(D,L-1);
    for(int l=0;l<L;l++){
      mu.col(l) = weighted_colMeans(Y,Z.col(l),tau);
    }
    Sigma = sigma_updata(Y,Z,mu,L,nu,Lambda);
    ll_hist(h) = L1[1];
  }
  L1 = Ecalc(Y,X,beta,mu,Sigma,N,L,D);
  return Rcpp::List::create(Rcpp::Named("beta")=beta,
                            _["mu"]=mu,_["Sigma"]=Sigma,
                            _["loglik"]=ll_hist,
                            _["W"]=L1[0]);
}

// [[Rcpp::export]]
Rcpp::List mixtruncGaussEM(const arma::mat & Y, const arma::mat & X, const int & L,
                        const arma::mat & muini,const arma::mat & betaini, const arma::cube & Sigmaini,const arma::mat & wini,
                        const double & tau, const double & nu, const arma::mat & Lambda, const int & num_iter){
  arma::mat mu = muini;
  arma::mat beta = betaini;
  arma::cube Sigma = Sigmaini;
  int N = Y.n_rows;
  int D = X.n_cols;
  arma::mat pre_Z = Y;
  arma::mat W = wini;
  arma::vec llhist(num_iter-1);
  llhist.fill(0);
  List LZ(2);
  List beta_grad(2);
  arma::mat hessian(D*(L-1),D*(L-1));
  for(int h=1;h<num_iter;h++){
    if(Sigma.has_nan()){
      break;
    }
    arma::cube invSigma = Sigma;
    for(int l=0; l<L; l++){
      invSigma.slice(l) = arma::inv(Sigma.slice(l));
    }
    LZ = simZ(pre_Z,mu,invSigma,L,W);
    arma::mat Z = LZ[0];
    W = simW(Z,X,beta,mu,invSigma,L);
    llhist(h-1) = LZ[1];
    beta_grad = dQ1_d2Q1(beta,X,W,D,L,N);
    arma::mat dbeta = beta_grad[0];
    arma::mat d2beta = beta_grad[1];
    hessian = d2beta;
    beta = arma::vectorise(beta,0) - arma::pinv(d2beta)*arma::vectorise(dbeta,0);
    beta.reshape(D,L-1);
    for(int l=0;l<L;l++){
      mu.col(l) = weighted_colMeans(Z,W.col(l),tau);
    }
    Sigma = sigma_updata(Z,W,mu,L,nu,Lambda);
    pre_Z = Z;
  }
  return Rcpp::List::create(Rcpp::Named("beta")=beta,
                            _["mu"]=mu,_["Sigma"]=Sigma,
                            _["W"]=W,_["loglik"]=llhist);
}
