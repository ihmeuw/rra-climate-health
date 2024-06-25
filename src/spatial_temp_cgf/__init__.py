from ctypes import cdll
# Lmer needs to know where to find libstdc++.so.6
cdll.LoadLibrary('/mnt/share/homes/victorvt/envs/cgf_temperature/lib/libstdc++.so.6')
