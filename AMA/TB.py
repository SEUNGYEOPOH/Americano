#!/usr/bin/env python
# coding: utf-8

# In[14]:


class get_start:
    import pandas as pd
    import numpy as np
    import pycaret.regression as reg
    import pycaret.classification as cls
    def __init__(self, data, categori, target ,use_gpu):
        self.categori = categori
        self.taget = target
        self.use_gpu = use_gpu
        self.data = data
            
    def pre_process(self):
        if self.categori == "regression":

            return reg.setup(self.data, target = self.taget, session_id = 123, use_gpu=self.use_gpu)
        elif self.categori =="classification":
            return cls.setup(self.data, target = self.taget, session_id = 123, use_gpu=self.use_gpu)
        
    def compare_models(s):
        if self.categori == "regression":
            return reg.compare_models()
        elif self.categori == "classification":
            return cls.compare_models


# In[ ]:




