# TP0, with a modified loss function


## First, the simple thing

Weight all channels equally, average across channels.

## Second, the more complicated thing

Note: I never pursued this because the simple thing did what we wanted.
That is, it redistributed the loss to give slightly less attention to the 2D
variables and slightly more to the 3D variables.

The main idea here is that I think it's weird that e.g. temperature is treated as a
separate variable from 2m temperature in the loss function.
So I am trying the following:

* gather all variables of the same type, average them together. So e.g. tmp2m is
  treated as a layer of temperature, basically.
  Same for specific humidity, and velocities.
* Average all these number of variables together
