# Data

`samples/` contains one test trajectory per problem for smoke testing. Full
training, validation, and test datasets are supplied as release assets and can
also be regenerated deterministically from the production configurations.

All saved states are conservative cell averages. Higher-CFL training pairs are
formed by integer temporal subsampling; the reference solvers themselves retain
their stated small internal time steps.
