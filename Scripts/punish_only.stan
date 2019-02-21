// The 'data' block list all input variables that are given to Stan from R. You need to specify the size of the arrays
data {
  int ntrials;  // number of trials per participant; 'int' means that the values are integers
  int nsub;     // number of subjects
  int plus[ntrials,nsub];     // CS plus expectancy ratings associated with punishment (scream) 75 % contingency
  int minus[ntrials,nsub];    // CS minus expectancy ratings associated with no punishment 100% of the time
  int includeTrial[ntrials];     // whether the data from this trial should be fitted (0 for trials to exclude) not currently using this. 
}


// The 'parameters' block defines the parameter that we want to fit
parameters {
  // Stan syntax explanation:
  // real : parameters are real numbers
  // <lower=0,upper=1> : parameter is in the range of 0 to 1
  // alpha : name of the parameter
  // [nsub,2] : size of the parameter (number of rows, number of columns)
  // Group level parameters
  real<lower=0,upper=1> alpha_mu; // group level learning rate mean - pos neg
  real<lower=0> alpha_sd; // group level learning rate sd
  real<lower=0> beta_mu; // group level mean for temperature
  real<lower=0> beta_sd; // group level sd for temperature
  // Single subject parameters
  real<lower=0,upper=1> alpha[nsub]; // learning rate - separate learning rates for positive and negative
  real<lower=0> beta[nsub] ;   // temperature (i.e. how consistent choices are); one per participant
}

// This block runs the actual model
model {
  // temporary variables that we will compute for each person and each trial
  real QA[ntrials,nsub];  //Q value of shape A
  real QB[ntrials,nsub]; // Q value of shape B
  real deltaA[ntrials-1,nsub]; // prediction error for shape A
  real deltaB[ntrials-1,nsub];    // prediction error for shape B


  // Priors
  // as no prior for alpha is defined, it implicitly becomes the range it is given in the parameters block, i.e. from 0 to 1
  //  betawin1 ~ normal(0,1); //made this 10 as generally larger in this task

  // Priors for the individual subjects are the group:
  for (p in 1:nsub){
    alpha[p] ~ normal(alpha_mu,alpha_sd);
    beta[p]  ~ normal(beta_mu,beta_sd);
  }


  // The learning model: the aim is to define how the input data (i.e. the reward outcomes, the reward magnitudes) and parameters relate to the behavior
  // The basic structure of the model is exactly as in Matlab before:
  // The first lines define the learning of reward probabilities, then these are combined with magnitudes to give utilities
  // Then the choice utilities are linked to the actual choice using a softmax function
  for (p in 1:nsub){ // run the model for each subject
    // Learning
    QA[1,p] = 0; // first trial, best guess is that values are at 0
    QB[1,p] = 0;
    for (t in 1:ntrials-1){
      deltaA[t,p] = (plus[t,p]) - QA[t,p]; // prediction error for A
      deltaB[t,p] = (1-plus[t,p])- QB[t,p]; // prediction error for B
      QA[t+1,p] = QA[t,p] + alpha[p] * deltaA[t,p]; // Q learning for A
      QB[t+1,p] = QB[t,p] + alpha[p] * deltaB[t,p]; // should delta be same in both cases?
    }


    // Decision - combine predictions of punish probability with magnitudes
    for (t in 1:ntrials){
      if (includeTrial[t]==1){ // if  we want to fit the trial (we don't have missing responses)

        // Compare the choice probability (based on the utility) to the actual choice
        // See the handout for the syntax of the bernoulli_logit function
        // equivalently we could have written (as we have done previously in Matlab; but this runs a bit less well in Stan).:
        // ChoiceProbability1[it,is] = 1/(1+exp(beta[is]*(util2[it,is]-util1[it,is]))); // the softmax is an 'inv_logit'
        // opt1Chosen[it,is] ~ bernoulli(ChoiceProbability1[it,is]);
        // choices[t,p] ~ (exp(QA[t,p]/beta[p])/((exp(QA[t,p]/beta[p])+(QB[t,p]/beta[p]); // could do using bernoulli_logit
        // need to decide what distribution works best here with expectancy ratings
         rating[t,p] ~ bernoulli(exp(QA[t,p]/beta[p])/(exp(QA[t,p]/beta[p])+exp(QB[t,p]/beta[p])));
      }
    }
  }
}
