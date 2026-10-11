"""CPU tests for time-aligned clearance and unmasked ranking arithmetic."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from common import BASE, save

spec=importlib.util.spec_from_file_location('route_measure',BASE/'measure.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
robot=SimpleNamespace(px=0.,py=0.,radius=0.)
human=np.array([[.5,1.,0.,-4.,0.]])
value=m.swept(robot,human,np.array([[4.,0.]]),human[:,2:4])[0]
assert abs(value-np.sqrt(.125))<1e-12
stationary=m.swept(robot,np.array([[1.,0.,0.,0.,0.]]),np.array([[0.,0.]]),np.zeros((1,2)))[0]
assert stationary==1.
equal_velocity=m.swept(robot,np.array([[1.,0.,1.,0.,0.]]),np.array([[1.,0.]]),np.array([[1.,0.]]))[0]
assert equal_velocity==1.
metrics=m.metrics(np.array([.6,.6,.5,.4,.3,.2]),np.array([.1,.2,.8,.4,.3,.2]),
                  {'endpoint':np.array([.1,.2,.3,.4,.5,.6])})
assert metrics['v_gap']==0. and metrics['value_top5']==[0,1,2,3,4]
assert abs(metrics['raw_gap']-.4)<1e-12
assert abs(metrics['endpoint']['spread']-.4)<1e-12
out=dict(status='已验证通过',checks=['time-aligned crossing, not line-segment intersection',
    'zero relative velocity','equal nonzero velocities','unmasked V ties independent of raw-score ranking'],
    asynchronous_crossing_clearance=float(value),training=False)
save(BASE/'metric-unit-checks.json',out)
print(out)
