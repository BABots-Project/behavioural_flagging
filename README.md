# Behavioural Flagging

The data used is from Broekmans et al 2016 (https://datadryad.org/stash/dataset/doi:10.5061/dryad.t0m6p)

The behaviour of a worm is characterised by crawls alteranted by reorientations as per Salvador et al 2014. The current implementation allows to recognize the reorientations:
- omegas, where the solidity of the worm is used
- reversals, an adaptation of Hardaker's method to detect reversal events
- pauses, characterised by a significant decrease in velocity (<=1/3 of the average)
- pirouettes, characterised by an omega followed by a reversal within 0.5s

Next step is the implementation of the crawl recognition. The algorithm will produce a final list of all states as they occur starting from the images and centroid coordinates of a tracked worm.
