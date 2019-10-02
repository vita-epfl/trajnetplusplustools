Tools
=====

* summary table and plots: ``python -m trajnettools.summarize <dataset_files>``
* plot trajectories in a scene: ``python -m trajnettools.trajectories <dataset_file>``
* visualize interactions: ``python -m trajnettools.visualize_interactions <dataset_file> --interaction_type 'ca'``

APIs
====

* ``trajnettools.Reader``: class to read the dataset_file
* ``trajnettools.show``: module containing contexts for visualizing ``rows`` and ``paths``
* ``trajnettools.writers``: write a trajnet dataset file
* ``trajnettools.metrics``: contains ``average_l2(), final_l2() and collision()`` functions


Dataset
=======

Datasets are split into ``train``, ``val`` and ``test`` set.
Every line is a self contained JSON string (ndJSON_).

Scene:

.. code-block:: json

    {"scene": {"id": 266, "p": 254, "s": 10238, "e": 10358, "fps": 2.5, "tag": 2}}

Track:

.. code-block:: json

    {"track": {"f": 10238, "p": 248, "x": 13.2, "y": 5.85}}

with:

* ``id``: scene id
* ``p``: pedestrian id
* ``s``, ``e``: start and end frame id
* ``fps``: frame rate
* ``tag``: trajectory type
* ``f``: frame id
* ``x``, ``y``: x- and y-coordinate in meters
* ``pred_number``: (optional) prediction number for multiple output predictions
* ``scene_id``: (optional) corresponding scene_id for multiple output predictions

Frame numbers are not recomputed. Rows are resampled to about
2.5 rows per second.


Dev
===

.. code-block:: sh

    pylint trajnettools
    python -m pytest
    # optional: mypy trajnettools --disallow-untyped-defs


Dataset Summaries
=================

biwi_hotel:

+----------------------------------------------------+----------------------------------------------------+
| .. image:: docs/train/biwi_hotel.ndjson.theta.png  | .. image:: docs/train/biwi_hotel.ndjson.speed.png  |
+----------------------------------------------------+----------------------------------------------------+

crowds_students001:

+-----------------------------------------------------------+-----------------------------------------------------------+
| .. image:: docs/train/crowds_students001.ndjson.theta.png | .. image:: docs/train/crowds_students001.ndjson.speed.png |
+-----------------------------------------------------------+-----------------------------------------------------------+

crowds_students003:

+-----------------------------------------------------------+-----------------------------------------------------------+
| .. image:: docs/train/crowds_students003.ndjson.theta.png | .. image:: docs/train/crowds_students003.ndjson.speed.png |
+-----------------------------------------------------------+-----------------------------------------------------------+

crowds_zara02:

+-----------------------------------------------------------+-----------------------------------------------------------+
| .. image:: docs/train/crowds_zara02.ndjson.theta.png      | .. image:: docs/train/crowds_zara02.ndjson.speed.png      |
+-----------------------------------------------------------+-----------------------------------------------------------+

crowds_zara03:

+-----------------------------------------------------------+-----------------------------------------------------------+
| .. image:: docs/train/crowds_zara03.ndjson.theta.png      | .. image:: docs/train/crowds_zara03.ndjson.speed.png      |
+-----------------------------------------------------------+-----------------------------------------------------------+

dukemtmc:

+-----------------------------------------------------------+-----------------------------------------------------------+
| .. image:: docs/train/dukemtmc.ndjson.theta.png           | .. image:: docs/train/dukemtmc.ndjson.speed.png           |
+-----------------------------------------------------------+-----------------------------------------------------------+

syi:

+-----------------------------------------------------------+-----------------------------------------------------------+
| .. image:: docs/train/syi.ndjson.theta.png                | .. image:: docs/train/syi.ndjson.speed.png                |
+-----------------------------------------------------------+-----------------------------------------------------------+

wildtrack:

+-----------------------------------------------------------+-----------------------------------------------------------+
| .. image:: docs/train/wildtrack.ndjson.theta.png          | .. image:: docs/train/wildtrack.ndjson.speed.png          |
+-----------------------------------------------------------+-----------------------------------------------------------+

Interactions
============

leader_follower:

+--------------------------------------------------------+-----------------------------------------------------------+
| .. image:: docs/train/crowds_zara02.ndjson_1_9.png     | .. image:: docs/train/crowds_zara02.ndjson_1_9_full.png   |
+--------------------------------------------------------+-----------------------------------------------------------+

collision_avoidance:

+---------------------------------------------------------+------------------------------------------------------------+
| .. image:: docs/train/crowds_zara02.ndjson_2_25.png     | .. image:: docs/train/crowds_zara02.ndjson_2_25_full.png   |
+---------------------------------------------------------+------------------------------------------------------------+

group:

+--------------------------------------------------------+-----------------------------------------------------------+
| .. image:: docs/train/crowds_zara02.ndjson_3_9.png     | .. image:: docs/train/crowds_zara02.ndjson_3_9_full.png   |
+--------------------------------------------------------+-----------------------------------------------------------+

others:

+---------------------------------------------------------+------------------------------------------------------------+
| .. image:: docs/train/crowds_zara02.ndjson_4_13.png     | .. image:: docs/train/crowds_zara02.ndjson_4_13_full.png   |
+---------------------------------------------------------+------------------------------------------------------------+

.. _ndJSON: http://ndjson.org/
