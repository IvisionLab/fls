#!/usr/bin/env python3
#%%
# concat results from HOG+SVM
import json
import datetime

# ssiv bahia results filepath
SSIV_BAHIA_FILEPATH = "assets/results/hog_svm_ssiv_bahia_20181211T2249.json"

# jequitaia results filepath
JEQUITAIA_FILEPATH = "assets/results/hog_svm_jequitaia_20181211T2150.json"

# balsa results filepath
BALSA_FILEPATH = "assets/results/hog_svm_balsa_20181211T2234.json"

with open(SSIV_BAHIA_FILEPATH) as jsonfile:
  ssiv_bahia_results = json.load(jsonfile)

with open(JEQUITAIA_FILEPATH) as jsonfile:
  jequitaia_results = json.load(jsonfile)

with open(BALSA_FILEPATH) as jsonfile:
  balsa_results = json.load(jsonfile)


#%%
# concat results
all_results = ssiv_bahia_results + jequitaia_results + balsa_results

result_filepath = "hog_svm_all_{:%Y%m%dT%H%M}.json".format(datetime.datetime.now())
json.dump(all_results, open(result_filepath, 'w'))

