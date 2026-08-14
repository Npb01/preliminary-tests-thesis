These runs were labelled '_balanced' but the loss rescaling never took
effect: train_model passed balance_losses=False / scale_ttne=1.0 /
scale_rrt=1.0 as literals to train_epoch instead of forwarding its own
arguments. Each folder here was verified to be metric-identical to the
corresponding UNBALANCED run at the same seed, i.e. it is a duplicate,
not a control. Excluded from analysis for that reason. Nothing here is
usable as evidence about loss rebalancing.
