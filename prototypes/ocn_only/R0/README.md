# Ocn-only run
This is the default ocean-only configuration with full 27 years of REPLAY data
on 6 hours of time stepping with two time steps as inputs. Note that this
configuration is a revamped version where more forcing from the atmosphere is
used to mimick the physical models.  Below is the updated configuration:

## Configuration
* Forcing: U10m, V10m, LW/SW radiation fluxes into the ocean, T_lml, spfh2m,
* spfh_lml Prognostic: SSH, Temp, Salinity, U/V, Static/Clock: land-sea mask,
* day/year progress

Note that there is no bathymetry at this point. This would be added in the
future and will be made a separate run.

## Outcome
