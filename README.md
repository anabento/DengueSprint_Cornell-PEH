# DengueSprint_Cornell-PEH

## Team and Contributors

Cornell University, College of Veterinary Medicine, Department of Public and Ecosystem Health, Bento Lab

- Tijs Alleman
- Ana Bento
- Laura Alexander

## Repository structure

```
├── src
|  |── ...
|  └── Model source code.
└── data
  |── raw
  |   └── Unaltered data.
  └── interim 
      └── Converted or altered data.
```

## Libraries and Dependencies

Setup/update the `environment`: All dependencies needed to run the scripts are collected in the conda `DENGUE_SPRINT.yml` file. To set up the environment,

  ```bash
  conda env create -f DENGUE_SPRINT.yml
  conda activate DENGUE_SPRINT
  ```

Install the `epiweeks` package into the conda environment.

  ```bash
  conda activate DENGUE_SPRINT
  pip install epiweeks
  ```

## Data and Variables

Only the `dengue.csv` file made available by the mosqlimate project was used. This file contains the weekly DENV incidence per epiweek and per Brazilian municipality from 2010-2025.

## Model Training

The model presents a simple baseline model which we will use to benchmark future DENV modeling efforts.

1. The raw data (`dengue.csv`) were aggregated to the level of the 27 Brazilian federative units.
2. The data from 2010-2025 were grouped by week number (1-53), resulting in a sample consiting of 15 DENV incidence counts per week number and per Brazilian UF. Let `I_{i,j}` denote the distribution of DENV incidence in UF `i` during week `j` across the available training seasons.
3. A negative binomial distribution with parameters `r_{i,j}` (number of succeses) and `p_{i,j}` (probability of succes) was fit to the empirical distribution `I_{i,j}`.
4. The predictive distribution `I^{*}_{i,j} ~ NegBinom(r_{i,j}, p_{i,j})` was then used to generate the forecasted quantiles for submission.

## References

NA.

## Data Usage Restrictions

Only data up to EW25 were used to generate predictions from EW 41 of the same year to EW 40 of the next year.

## Predictive uncertainty

The fitted negative binomial was used to generate quantiles using the `scipy.nbinom.ppf()` function.