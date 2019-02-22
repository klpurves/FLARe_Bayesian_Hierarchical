
// The 'data' block list all input variables that are given to Stan from R. You need to specify the size of the arrays
data {
  int ntrials;  // number of trials per participant; "int" means that the values are integers
  int nsub;     // number of subjects
  int rewardA[ntrials,nsub];     // if rewarded when chose shape A
  int punishA[ntrials,nsub];     // if lost when chose shape A
  int choices[ntrials,nsub];     // if chose shape A
  int includeTrial[ntrials];     // whether the data from this trial should be fitted (0 for trials to exclude)
}


// The 'parameters' block defines the parameter that we want to fit
parameters {
  // Stan syntax explanation:
  // real : parameters are real numbers
  // <lower=0,upper=1> : parameter is in the range of 0 to 1
  // alpha : name of the parameter
  // [nsub,2] : size of the parameter (number of rows, number of columns)
  // Group level parameters
  real<lower=0,upper=1> alpha_mu[2]; // group level learning rate mean - pos neg
  real<lower=0> alpha_sd[2]; // group level learning rate sd
  real<lower=0> beta_mu[2]; // group level mean for temperature
  real<lower=0> beta_sd[2]; // group level sd for temperature
  // Single subject parameters
  real<lower=0,upper=1> alpha[nsub,2]; // learning rate - separate learning rates for positive and negative
  real<lower=0> beta[nsub,2] ;   // temperature (i.e. how consistent choices are); one per participant
}

// This block runs the actual model
model {
  // temporary variables that we will compute for each person and each trial
  real QA[ntrials,nsub];  //Q value of shape A
  real QB[ntrials,nsub]; // Q value of shape B
  real deltaA[ntrials-1,nsub]; // prediction error for shape A
  real deltaB[ntrials-1,nsub];    // prediction error for shape B
  real ProbA[ntrials,nsub]; // probability of choosing shape A


  // Priors
  // as no prior for alpha is defined, it implicitly becomes the range it is given in the parameters block, i.e. from 0 to 1
  //  betawin1 ~ normal(0,1); //made this 10 as generally larger in this task

  // Priors for the individual subjects are the group:
  for (p in 1:nsub){
    alpha[p,1] ~ normal(alpha_mu[1],alpha_sd[1]);
    alpha[p,2] ~ normal(alpha_mu[2],alpha_sd[2]);
    beta[p,1]  ~ normal(beta_mu[1],beta_sd[1]);
    beta[p,2]  ~ normal(beta_mu[2],beta_sd[2]);
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
      deltaA[t,p] = (rewardA[t,p]-punishA[t,p]) - QA[t,p]; // prediction error for A
      deltaB[t,p] = ((1-rewardA[t,p])-(1-punishA[t,p])) - QB[t,p]; // prediction error for B
      QA[t+1,p] = QA[t,p] + rewardA[t,p] *  alpha[p,1] * deltaA[t,p] + punishA[t,p] * alpha[p,2] * deltaA[t,p]; // Q learning for A
      QB[t+1,p] = QB[t,p] + (1-rewardA[t,p]) *  alpha[p,1] * deltaB[t,p] + (1-punishA[t,p]) * alpha[p,2] * deltaB[t,p]; // should delta be same in both cases?
    }


    // Decision - combine predictions of reward probability with magnitudes
    for (t in 1:ntrials){
      if (includeTrial[t]==1){ // if  we want to fit the trial (we don't have missing responses)

        // Compare the choice probability (based on the utility) to the actual choice
        // See the handout for the syntax of the bernoulli_logit function
        // equivalently we could have written (as we have done previously in Matlab; but this runs a bit less well in Stan).:
        // ChoiceProbability1[it,is] = 1/(1+exp(beta[is]*(util2[it,is]-util1[it,is]))); // the softmax is an 'inv_logit'
        // opt1Chosen[it,is] ~ bernoulli(ChoiceProbability1[it,is]);
        // here the 1 and 2 corresponds to beta reward and beta loss
        // choices[t,p] ~ (exp(QA[t,p]/beta[p])/((exp(QA[t,p]/beta[p])+(QB[t,p]/beta[p]); // could do using bernoulli_logit
         choices[t,p] ~ bernoulli(rewardA[t,p]*exp(QA[t,p]/beta[p,1])/(exp(QA[t,p]/beta[p,1])+exp(QB[t,p]/beta[p,1]))
         +punishA[t,p]*exp(QA[t,p]/beta[p,2])/(exp(QA[t,p]/beta[p,2])+exp(QB[t,p]/beta[p,2])));
      }
    }
  }
}

generated quantities { //does the same calculations again for the fitted values
  real loglik[nsub];
  real QA[ntrials,nsub];  //Q value of shape A
  real QB[ntrials,nsub]; // Q value of shape B
  real deltaA[ntrials-1,nsub]; // prediction error for shape A
  real deltaB[ntrials-1,nsub];    // prediction error for shape B

//this code is basically a copy of the model block
  for (p in 1:nsub){
    loglik[p] = 0; //initialise at 0
    QA[1,p] = 0;
    QB[1,p] = 0;
    for (t in 1:ntrials-1){
      deltaA[t,p] = (rewardA[t,p]-punishA[t,p]) - QA[t,p]; // prediction error for A
      deltaB[t,p] = ((1-rewardA[t,p])-(1-punishA[t,p])) - QB[t,p]; // prediction error for B
      QA[t+1,p] = QA[t,p] + rewardA[t,p] *  alpha[p,1] * deltaA[t,p] + punishA[t,p] * alpha[p,2] * deltaA[t,p]; // Q learning for A
      QB[t+1,p] = QB[t,p] + (1-rewardA[t,p]) *  alpha[p,1] * deltaB[t,p] + (1-punishA[t,p]) * alpha[p,2] * deltaB[t,p]; // should delta be same in both cases?
    }


    for (t in 1:ntrials){
      if (includeTrial[t]==1){ // if  we want to fit the trial (we don't have missing responses)

      // increments the log likelihood trial by trial using the log choice prob and parameters estimated in the model block
         loglik[p] =  loglik[p] + log((rewardA[t,p]*exp(QA[t,p]/beta[p,1])/(exp(QA[t,p]/beta[p,1])+exp(QB[t,p]/beta[p,1]))
         +punishA[t,p]*exp(QA[t,p]/beta[p,2])/(exp(QA[t,p]/beta[p,2])+exp(QB[t,p]/beta[p,2]))));
      }
    }
  }
}
