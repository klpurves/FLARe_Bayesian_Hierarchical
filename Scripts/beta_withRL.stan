data {
  int nsub; //number of participants
  int ntrials; //per stimulus
  real screamPlus[ntrials,nsub]; //scream CS+
  real screamMinus[ntrials,nsub]; //scream CS-
  real ratingsPlus[ntrials,nsub]; //rating per sub per trial 0-1
  real ratingsMinus[ntrials,nsub]; //rating per sub per trial 0-1
}

parameters {
  real <lower=0,upper=1>alpha[nsub]; //learning rate
  real <lower=0> beta[nsub]; //calculate the beta in the beta distribution
}

model {
  real VPlus[ntrials,nsub]; // value CS+
  real VMinus[ntrials,nsub]; // value CS-
  real deltaPlus[nsub,ntrials-1]; // prediction error for  CS+
  real deltaMinus[nsub,ntrials-1];    // prediction error for CS-


  for (p in 1:nsub){
    VPlus[1,p]=0.5;
    VMinus[1,p]=0.5;
    for (t in 1:(ntrials-1)){
      deltaPlus[t,p] = screamPlus[t,p]-VPlus[t,p]; // prediction error calc CS+
      deltaMinus[t,p] = screamMinus[t,p]-VMinus[t,p]; // ditto CS-
      VPlus[t+1,p]=VPlus[t,p]+alpha[p]*deltaPlus[t,p]; // value calc CS+
      VMinus[t+1,p]=VMinus[t,p]+alpha[p]*deltaPlus[t,p]; // ditto CS-
    }

    for (t in 1:ntrials){
      ratingsPlus[t,p] ~ beta(VPlus[t,p],beta[p]);
      ratingsMinus[t,p] ~ beta(VMinus[t,p],beta[p]);
    }
  }
}
