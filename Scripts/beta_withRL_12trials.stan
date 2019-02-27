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
  real deltaPlus[nsub,ntrials]; // prediction error for  CS+
  real deltaMinus[nsub,ntrials];    // prediction error for CS-


  for (p in 1:nsub){
    VPlus[1,p]=0.5;   // assume .5 valence with no experience before trial 1 occurs.
    VMinus[1,p]=0.5;   // ditto

    deltaPlus[1,p]=0.5;  // trial 1 is always reinforced, so can say delta will always be 0.5
    deltaMinus[1,p]=-0.5;  //trial 1 is never reinforced, so csan say delta will always be -0.5

    for (t in 2:(ntrials)){    // need prediction error for each trial (ie every one of 12 trials is reinforced or not)
      deltaPlus[t,p] = screamPlus[t,p]-VPlus[t,p]; // prediction error calc CS+
      deltaMinus[t,p] = screamMinus[t,p]-VMinus[t,p]; // ditto CS-
      VPlus[t,p]=VPlus[t-1,p]+alpha[p]*deltaPlus[t,p]; // value calc CS+
      VMinus[t,p]=VMinus[t-1,p]+alpha[p]*deltaMinus[t,p]; // ditto CS-
    }

    for (t in 1:ntrials){
      ratingsPlus[t,p] ~ beta(VPlus[t,p],beta[p]);
      ratingsMinus[t,p] ~ beta(VMinus[t,p],beta[p]);
    }
  }
}
