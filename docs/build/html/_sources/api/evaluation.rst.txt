Evaluation
==========

**A** - Point Cloud Based Metrics
---------------------------------

*a*. Chamfer Distance
^^^^^^^^^^^^^^^^^^^^^

:math:`D_{C}(P, P^*) = \frac{1}{|P|} \sum_{p \in P} d(p, P^*)^2 + \frac{1}{|P'|} \sum_{p^* \in P^*} d(p^*, P)^2`

.. autofunction:: scene_point_etk.evaluation.chamfer_distance


*b*. Hausdorff Distance
^^^^^^^^^^^^^^^^^^^^^^^

:math:`D_{H}(P, P^*) = \max \left\{ \max_{p \in P} d(p, P^*), \max_{p^* \in P^*} d(p^*, P) \right\}`

.. autofunction:: scene_point_etk.evaluation.hausdorff_distance

*c*. Modified Hausdorff Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:math:`D_{MH}(P, P^*) = \max \left\{ \frac{1}{|P|} \sum_{p \in P} d(p, P^*), \frac{1}{|P^*|} \sum_{p^* \in P^*} d(p^*, P) \right\}`

.. autofunction:: scene_point_etk.evaluation.modified_hausdorff_distance

*d*. Median Point Distance
^^^^^^^^^^^^^^^^^^^^^^^^^^

:math:`D_{MP}(P, P^*) = \max \left\{ \underset{p \in P}{\text{median}}\ d(p, P^*), \underset{p^* \in P^*}{\text{median}}\ d(p^*, P) \right\}`

.. autofunction:: scene_point_etk.evaluation.median_point_distance

