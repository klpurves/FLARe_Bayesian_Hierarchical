data {
  int nsub; //number of participants
  int ntrials; //per stimulus
  real nothingPlus[nsub]; //number of times nothing happened CS+
  real nothingMinus[nsub]; //number of times nothing happened CS-
  real ratingsPlus[nsub,ntrials]; //rating per sub per trial 0-1
  real ratingsMinus[nsub,ntrials]; //rating per sub per trial 0-1
}

parameters {
  real <lower=0> beta[nsub]; //calculate the beta in the beta distribution
  real <lower=0> scaling[nsub];
}

model {
  real alphaPlus[nsub];
  real alphaMinus[nsub];

  for (p in 1:nsub){
    alphaPlus[p] =  scaling[p]*nothingPlus[p]/ntrials;
    alphaMinus[p] =  scaling[p]*nothingMinus[p]/ntrials;
    ratingsPlus[p,] ~ beta(alphaPlus[p],beta[p]);
    ratingsMinus[p,] ~ beta(alphaMinus[p],beta[p]);
  }
}
