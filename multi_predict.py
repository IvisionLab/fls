#%% [markdown]
# ## Rboxnet - Test prediction
#

import numpy as np
from predict import predict


min_noise = 0.05
max_noise = 0.5
noise_step = max_noise/5.0
noises = np.arange(min_noise, max_noise+noise_step, noise_step)

for noise in noises:
  predict(noise)