"""
Methodology types of the pollsters (``mtype`` of the catalogue) that stay out of the model.

It imports nothing, so the model and the web layer share it without loading each other.
"""

# Aggregators (averages of other polls) and online panels: their polls never enter the average, they get no rating
# and they are left out of the public rankings and listings; their errors and their means are still measured
DROP_MTYPES = ('aggr', 'online')
