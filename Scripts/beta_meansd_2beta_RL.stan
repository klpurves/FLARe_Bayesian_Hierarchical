data {
  int nsub; //number of participants
  int ntrials; //per stimulus
  real screamPlus[ntrials,nsub]; //scream CS+
  real screamMinus[ntrials,nsub]; //scream CS-
  real ratingsPlus[ntrials,nsub]; //rating per sub per trial 0-1
  real ratingsMinus[ntrials,nsub]; //rating per sub per trial 0-1
  real cdf_scale; // scaled 0.5 for adjusting from discrete values in beta dist
}

parameters {
  real <lower=0,upper=1>alpha[nsub]; //learning rate
  real <lower=0> beta[nsub,2]; //calculate distribution variance.
                                          //Basically how confident they are when rating
                                          // related to uncertainty possibly?
                                          // to add 2, beta[nsub,2] and where used [p,1] or [p,2] by shape
  vector<lower=0.5-cdf_scale,upper=0.5+cdf_scale>[nsub] first;
}


model {
  real shape1_Plus[ntrials,nsub]; //shape parameter 1 CS+
  real shape1_Minus[ntrials,nsub]; // shape parameter 1 CS-
  real shape2_Plus[ntrials,nsub]; // shape paramter 2 CS+
  real shape2_Minus[ntrials,nsub]; // shape paramter 2 CS-


  matrix[ntrials,nsub] VPlus; // value CS+
  matrix[ntrials,nsub] VMinus; // value CS-

  real deltaPlus[ntrials-1,nsub]; // prediction error for  CS+
  real deltaMinus[ntrials-1,nsub];    // prediction error for CS-

  for (p in 1:nsub){
    VPlus[1,p]=first[p]; // assume that the mid point varies like our ratings. so a range between 0.5-0.0555556 and 0.5+0.0000056
    VMinus[1,p]=first[p];
    for (t in 1:(ntrials-1)){
      deltaPlus[t,p] = screamPlus[t,p]-VPlus[t,p]; // prediction error calc CS+
      deltaMinus[t,p] = screamMinus[t,p]-VMinus[t,p]; // ditto CS-
      VPlus[t+1,p]=VPlus[t,p]+alpha[p]*deltaPlus[t,p]; // value calc CS+
      VMinus[t+1,p]=VMinus[t,p]+alpha[p]*deltaMinus[t,p]; // ditto CS-
    }

    for (t in 1:ntrials){
      shape1_Plus[t,p] = VPlus[t,p] * ((VPlus[t,p] * (1-VPlus[t,p])) / beta[p,1]);
      shape1_Minus[t,p] = VMinus[t,p] * ((VMinus[t,p] * (1-VMinus[t,p])) / beta[p,2]);
      shape2_Plus[t,p] = (1-VPlus[t,p]) * ((VPlus[t,p] * (1-VPlus[t,p])) / beta[p,1]);
      shape2_Minus[t,p] = (1-VMinus[t,p]) * ((VMinus[t,p] * (1-VMinus[t,p])) / beta[p,2]);



      ratingsPlus[t,p] ~ beta(shape1_Plus[t,p],shape2_Plus[t,p]);
      ratingsMinus[t,p] ~ beta(shape1_Minus[t,p],shape2_Minus[t,p]);
    }
  }
}



// below is what will generate the log likelihoods.

generated quantities { //does the same calculations again for the fitted values
  real loglik[nsub];  // logliklihood paramter
  real shape1_Plus[ntrials,nsub]; //shape parameter 1 CS+
  real shape1_Minus[ntrials,nsub]; // shape parameter 1 CS-
  real shape2_Plus[ntrials,nsub]; // shape paramter 2 CS+
  real shape2_Minus[ntrials,nsub]; // shape paramter 2 CS-


  matrix[ntrials,nsub] VPlus; // value CS+
  matrix[ntrials,nsub] VMinus; // value CS-

  real deltaPlus[ntrials-1,nsub]; // prediction error for  CS+
  real deltaMinus[ntrials-1,nsub];    // prediction error for CS-

  for (p in 1:nsub){
    loglik[p]=0;

  {
    VPlus[1,p]=first[p]; // assume that the mid point varies like our ratings. so a range between 0.5-0.0555556 and 0.5+0.0000056
    VMinus[1,p]=first[p];
    for (t in 1:(ntrials-1)){
      deltaPlus[t,p] = screamPlus[t,p]-VPlus[t,p]; // prediction error calc CS+
      deltaMinus[t,p] = screamMinus[t,p]-VMinus[t,p]; // ditto CS-
      VPlus[t+1,p]=VPlus[t,p]+alpha[p]*deltaPlus[t,p]; // value calc CS+
      VMinus[t+1,p]=VMinus[t,p]+alpha[p]*deltaMinus[t,p]; // ditto CS-
    }

    for (t in 1:ntrials){
      shape1_Plus[t,p] = VPlus[t,p] * ((VPlus[t,p] * (1-VPlus[t,p])) / beta[p,1]);
      shape1_Minus[t,p] = VMinus[t,p] * ((VMinus[t,p] * (1-VMinus[t,p])) / beta[p,2]);
      shape2_Plus[t,p] = (1-VPlus[t,p]) * ((VPlus[t,p] * (1-VPlus[t,p])) / beta[p,1]);
      shape2_Minus[t,p] = (1-VMinus[t,p]) * ((VMinus[t,p] * (1-VMinus[t,p])) / beta[p,2]);

      // increments the log likelihood trial by trial using the log choice prob and parameters estimated in the model block
      loglik[p] += log(beta_cdf((ratingsPlus[t,p] + cdf_scale) , shape1_Plus[t,p],shape2_Plus[t,p]) -
      beta_cdf((ratingsPlus[t,p] - cdf_scale) , shape1_Plus[t,p],shape2_Plus[t,p])) +
      log(beta_cdf((ratingsMinus[t,p] + cdf_scale) , shape1_Minus[t,p],shape2_Minus[t,p]) -
      beta_cdf((ratingsMinus[t,p] - cdf_scale) , shape1_Minus[t,p],shape2_Minus[t,p]));
      }
    }
  }
}
