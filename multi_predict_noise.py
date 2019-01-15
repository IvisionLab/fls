#%% [markdown]
# ## Rboxnet - Test prediction
#

import numpy as np
from predict import predict, load_weights


min_noise = 0.0
max_noise = 0.2
noise_step = max_noise/10.0
noises = np.arange(min_noise, max_noise+noise_step, noise_step)
print(noises)
for noise in noises:
  load_weights()
  predict(noise)