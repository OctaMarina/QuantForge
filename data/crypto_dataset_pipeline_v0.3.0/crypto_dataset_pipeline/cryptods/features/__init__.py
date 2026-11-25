"""Feature families.  Every function is causal: the value at row t uses only information whose
availability time is <= the row's decision time (strictly < for other streams by default)."""
